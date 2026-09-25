"""Recurring schedules: the durable file, the skill sync, and what they dispatch.

Owns state/scheduled_tasks.json and the periodic reconciliation of skill-declared
schedules into it, then turns a schedule that is due into a queued task — skipping
any whose previous run is still pending or running — or, for a ``kind: "notify"``
row, into one owner notification with no model turn (the same table, the same
tick and the same lifecycle; only the dispatch verb differs).

The sync throttle is this module's own clock, not queue state: the writer and the
reader are both here.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, List
from ouroboros.consciousness_authority import apply_consciousness_authority
from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_resources
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug
from ouroboros.skill_loader import skill_identity_collision_names
from ouroboros.utils import atomic_write_json, in_worker_process, read_json_dict, utc_now_iso
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from supervisor.task_admission import record_scheduled_admission
from supervisor.schedule_time import (
    next_cron_time as _next_cron_time,
    once_due as _once_due,
    parse_schedule_time as _parse_schedule_time,
    prune_consumed_once_records as _prune_consumed_once,
    record_last_error as _record_last_error,
    schedule_next_run as _schedule_next_run,
    timezone_for_schedule as _timezone_for_schedule,
)




log = logging.getLogger(__name__)


def _queue():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import queue

    return queue


_last_skill_schedule_sync: float = 0.0


_SKILL_SCHEDULE_SYNC_INTERVAL_SEC: float = 60.0
# Per-thread depth of the held transactions, keyed by the TABLE's resolved lock
# path. Thread-local, so it needs no mutex of its own; the cross-thread ordering
# is the queue lock the transaction takes first.
_SCHEDULE_TX_STATE = threading.local()
_SCHEDULE_LOCK_SUFFIX = ".lock"
_SCHEDULE_LOCK_TIMEOUT_SEC = 8.0
_SCHEDULE_LOCK_STALE_SEC = 30.0
# The owner-governed actions over an EXISTING row; creation and editing stay on
# the upsert seam, so this set is exactly "what happens to a future dispatch".
SCHEDULE_ACTIONS: frozenset[str] = frozenset({"disable", "delete", "restore"})
# A skill row the owner disabled or deleted is retained with this marker: the
# lifecycle resync reconciles the row's CONTENT but may never re-arm it, and the
# marker survives a manifest edit because it is keyed by the same schedule id.
SUPPRESSED_OVERRIDES: frozenset[str] = frozenset({"disabled", "deleted"})
# Fields the RUNTIME owns. A caller builds its record from a GET that may already
# be stale, so these are always taken from the row on disk instead of the payload.
_RUNTIME_OWNED_FIELDS: tuple[str, ...] = (
    "source", "skill", "created_at", "last_run_at", "last_task_id", "last_error",
    "skill_content_hash", "manual_override",
)
# What an audit event may say about a row: lifecycle facts only. The task template
# is a private objective, never audit material, and would also be unbounded.
_AUDIT_ROW_KEYS: tuple[str, ...] = (
    "id", "name", "kind", "enabled", "source", "skill", "trigger", "timezone",
    "created_at", "updated_at", "last_run_at", "last_task_id", "completed_at",
    "next_run_at", "manual_override",
)
# The second dispatch verb of the one table: a ``kind: "notify"`` row emits one
# owner notification when due instead of enqueueing a task — no model turn, no
# task result; its receipt is the ``owner_notification`` events row itself. An
# absent ``kind`` (every row written before this verb existed) is a task row.
SCHEDULE_KIND_NOTIFY = "notify"
# ``manage_schedules`` is a model-facing tool.  Keep one page comfortably below
# the tool result cap even when a schedule table contains many rows or hostile
# (but valid) long strings.  The owner HTTP surface retains its own full rows.
_SCHEDULE_PAGE_DEFAULT = 12
_SCHEDULE_PAGE_MAX = 20
_SCHEDULE_FIELD_CHARS = 96
_SCHEDULE_OBJECTIVE_PREVIEW_CHARS = 200
# Keep the serialized model result below the 15k tool envelope while allowing
# the requested 20-row page to remain intact for ordinary lifecycle rows.
_QUEUE_SNAPSHOT_MAX_AGE_SEC = 300.0


class ScheduleStoreUnreadable(RuntimeError):
    """The durable schedule table exists but cannot be parsed.

    Every mutation refuses on this: rewriting the file from an empty document
    would silently erase every row the unreadable bytes still hold.
    """


class ScheduleLockTimeout(ScheduleStoreUnreadable, TimeoutError):
    """The table could not be reached within the lock bound; nothing was changed.

    Typed on both axes: a ``TimeoutError`` for the callers that already treat a
    missed lock as one, a ``ScheduleStoreUnreadable`` for the owner surfaces
    whose contract is "the state is unknown, the write was refused".
    """


class ScheduleRefused(RuntimeError):
    """A typed refusal from a schedule write (an owner-facing status + message)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = str(status)
        self.message = str(message)


def _scheduled_tasks_path(drive_root: pathlib.Path | None = None) -> pathlib.Path:
    return pathlib.Path(drive_root or _queue().DRIVE_ROOT) / _queue().SCHEDULED_TASKS_FILE


def _schedule_lock_key(lock_path: pathlib.Path) -> str:
    """Canonical per-table reentrancy key, including Windows case folding."""
    return os.path.normcase(os.path.realpath(str(lock_path)))


@contextlib.contextmanager
def schedule_transaction(drive_root: pathlib.Path | None = None):
    """Serialize a whole schedule read/modify/write across threads and processes.

    The transaction takes BOTH locks ITSELF — the queue lock first, the table's
    sidecar file lock second — so no caller can compose the pair in the other
    order. Asking callers to order them was the bug: ``schedule_followup`` wraps
    its cap-read and its write in one transaction, then ``upsert_scheduled_task``
    reached for the queue lock INSIDE that hold, against the scheduler tick's
    queue-then-table order. Owning the order here is what makes every caller
    provably consistent instead of individually audited.

    Reentrancy is keyed by the TABLE, not by a bare depth counter: a nested
    transaction on the same file rides the held file lock, while one addressing a
    different drive root still takes its own — a depth-only guard would let the
    second root's write run unserialized behind the first root's lock.
    """
    root = pathlib.Path(drive_root or _queue().DRIVE_ROOT)
    lock_path = _scheduled_tasks_path(root).with_name(
        _scheduled_tasks_path(root).name + _SCHEDULE_LOCK_SUFFIX)
    key = _schedule_lock_key(lock_path)
    held = getattr(_SCHEDULE_TX_STATE, "held", None)
    if held is None:
        held = {}
        _SCHEDULE_TX_STATE.held = held
    if held.get(key):
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    with _queue()._queue_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = acquire_exclusive_file_lock(
            lock_path, timeout_sec=_SCHEDULE_LOCK_TIMEOUT_SEC, stale_sec=_SCHEDULE_LOCK_STALE_SEC,
            owner_aware_stale=True)
        if fd is None:
            raise ScheduleLockTimeout(f"Could not acquire schedule lock {lock_path} within {_SCHEDULE_LOCK_TIMEOUT_SEC:g}s")
        held[key] = 1
        try:
            yield
        finally:
            held.pop(key, None)
            release_exclusive_file_lock(lock_path, fd)


def list_scheduled_tasks(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Return the persisted scheduled task table (a lenient READ projection)."""
    data = read_json_dict(_scheduled_tasks_path(drive_root)) or {}
    if not isinstance(data, dict):
        data = {}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        data["tasks"] = []
    data.setdefault("schema_version", 1)
    return data


def load_schedule_store(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """The same table, read under the rule a WRITER and an OWNER SURFACE need.

    ``list_scheduled_tasks`` renders an unreadable file as an empty table. That
    is wrong for a writer — the next atomic write would replace real rows with
    that emptiness — and wrong for a reader too, because "no schedules" is a
    claim, while an unparseable file means the state is UNKNOWN.

    Only an ABSENT path is a legitimate empty table (a store yet to be created).
    Everything else that is not a readable object-of-rows raises, and the cases
    stay distinguishable in the message because they need different owner
    responses: nothing there, something there that is not a regular file, a file
    the process cannot read, and bytes that do not parse. A row that is not an
    object raises too — the writers used to filter those out and then write the
    filtered list back, which is how a mutation about ONE schedule silently
    dropped another.
    """
    path = _scheduled_tasks_path(drive_root)
    try:
        # lstat, not exists(): a dangling symlink at the table's path IS present,
        # and calling it an absent store would let the next write follow it.
        path.lstat()
    except FileNotFoundError:
        return {"schema_version": 1, "tasks": []}
    except OSError as exc:
        raise ScheduleStoreUnreadable(
            f"{path} cannot be examined ({exc}); it is not a readable schedule "
            "table and will not be rewritten") from exc
    if not path.is_file():
        raise ScheduleStoreUnreadable(
            f"{path} is not a regular file, so it is not a readable schedule "
            "table; refusing to rewrite it")
    try:
        # Read the bytes here instead of through the lenient reader: it answers
        # None for "could not open" and for "could not parse" alike, and those
        # are different owner problems (a permission/IO fault vs a corrupt file).
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScheduleStoreUnreadable(
            f"{path} could not be read ({exc}); it is not a readable schedule "
            "table and will not be rewritten") from exc
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ScheduleStoreUnreadable(
            f"{path} is not a readable schedule table ({exc}); refusing to rewrite it") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tasks", []), list):
        raise ScheduleStoreUnreadable(
            f"{path} is not a readable schedule table; refusing to rewrite it")
    rows = list(data.get("tasks", []))
    malformed = [index for index, row in enumerate(rows) if not isinstance(row, dict)]
    if malformed:
        raise ScheduleStoreUnreadable(
            f"{path} is not a readable schedule table: {len(malformed)} row(s) are "
            f"not objects (first at index {malformed[0]}); refusing to rewrite it, "
            "because a write would drop them without saying so")
    data.setdefault("schema_version", 1)
    data["tasks"] = rows
    return data


def _is_consumed_once(record: Dict[str, Any]) -> bool:
    """A one-shot that already fired: a durable receipt, not a standing schedule."""
    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    return str(trigger.get("type") or "") == "once" and bool(record.get("completed_at"))


def _is_suppressed(record: Dict[str, Any]) -> bool:
    """A skill row the owner disabled or deleted and the resync may not re-arm."""
    return (str(record.get("source") or "") == "skill_manifest"
            and str(record.get("manual_override") or "").strip().lower() in SUPPRESSED_OVERRIDES)


def schedule_lifecycle_status(record: Dict[str, Any]) -> str:
    """The one lifecycle word every surface says about a row.

    ``consumed`` and ``suppressed`` are RETAINED history: neither dispatches
    again, and neither is an ``active`` schedule wearing a disabled flag.
    """
    if _is_consumed_once(record):
        return "consumed"
    if _is_suppressed(record):
        return "suppressed"
    return "active" if record.get("enabled", True) else "disabled"


def _bounded_projection_text(value: Any, limit: int = _SCHEDULE_FIELD_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _schedule_projection_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Project one durable row without exposing its unbounded task template."""
    row: Dict[str, Any] = {}
    # Keep only lifecycle facts useful to a model.  In particular, the durable
    # task template (including context/attachments) never crosses this seam.
    projection_keys = (
        "id", "name", "kind", "enabled", "source", "skill", "trigger",
        "created_at", "last_run_at", "last_task_id", "completed_at", "next_run_at",
    )
    for key in projection_keys:
        value = raw.get(key)
        if key == "trigger":
            trigger = value if isinstance(value, dict) else {}
            # Trigger fields are a closed, small shape.  Bound strings anyway so
            # a manually edited table cannot inflate a model-facing response.
            row[key] = {
                "type": _bounded_projection_text(trigger.get("type"), 32),
                **({"expr": _bounded_projection_text(trigger.get("expr"), _SCHEDULE_FIELD_CHARS)}
                   if trigger.get("expr") is not None else {}),
                **({"run_at": _bounded_projection_text(trigger.get("run_at"), _SCHEDULE_FIELD_CHARS)}
                   if trigger.get("run_at") is not None else {}),
            }
        elif key in {"id", "last_task_id", "skill"}:
            # Identities are selectors, not display previews. Truncating one can
            # address a different record on a later action or evidence read.
            row[key] = str(value) if value is not None else None
        elif isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
            row[key] = value
        else:
            row[key] = _bounded_projection_text(value)

    template = raw.get("task") if isinstance(raw.get("task"), dict) else {}
    notification = raw.get("notification") if isinstance(raw.get("notification"), dict) else {}
    objective_source = ""
    for candidate in (notification.get("text") if str(raw.get("kind") or "") == SCHEDULE_KIND_NOTIFY else None,
                      template.get("objective"), template.get("text"),
                      template.get("description"), raw.get("description"), raw.get("name")):
        if candidate is not None and str(candidate).strip():
            objective_source = str(candidate)
            break
    objective_truncated = len(objective_source) > _SCHEDULE_OBJECTIVE_PREVIEW_CHARS
    row["objective_preview"] = _bounded_projection_text(
        objective_source, _SCHEDULE_OBJECTIVE_PREVIEW_CHARS)
    # Keep both spellings during the contract migration: callers can discover
    # that the useful objective is only a preview without receiving ``task`` or
    # its potentially enormous context/attachments.
    row["objective_truncated"] = objective_truncated
    row["objective_preview_truncated"] = objective_truncated

    status = schedule_lifecycle_status(raw)
    row["status"] = status
    row["active"] = status == "active"
    row["consumed"] = status == "consumed"
    row["suppressed"] = status == "suppressed"
    # Retained rows are history the owner can still act on; only a suppressed
    # one can come back, and only after its skill is re-evaluated.
    row["retained"] = status in {"consumed", "suppressed"}
    row["restorable"] = status == "suppressed"
    return row


def schedule_tool_projection(
    data: Dict[str, Any], *, offset: int = 0, limit: int = _SCHEDULE_PAGE_DEFAULT,
    result_limit: int = 15_000,
) -> Dict[str, Any]:
    """Return a compact, bounded, paginated Activity/model schedule view.

    The persisted ``task`` template deliberately does not cross this boundary:
    context, attachments and metadata may be arbitrarily large.  Each row keeps
    a bounded objective preview and says whether it was truncated.  ``total``
    and ``next_offset`` make a partial page explicit to both the UI and a model.
    """
    source = data if isinstance(data, dict) else {}
    try:
        page_limit = max(1, min(_SCHEDULE_PAGE_MAX, int(limit)))
    except (TypeError, ValueError):
        page_limit = _SCHEDULE_PAGE_DEFAULT
    rows = [raw for raw in (source.get("tasks") or []) if isinstance(raw, dict)]
    total = len(rows)
    try:
        # Clamp an untrusted offset to the table so even a giant integer cannot
        # inflate the envelope while still making the page boundary explicit.
        page_offset = min(total, max(0, int(offset)))
    except (TypeError, ValueError):
        page_offset = 0
    schema_version = source.get("schema_version", 1)
    if not isinstance(schema_version, int):
        schema_version = 1
    page = rows[page_offset: page_offset + page_limit]
    projected = [_schedule_projection_row(raw) for raw in page]
    # Keep the complete envelope valid even when a caller asks for the maximum
    # page and the table contains unusually long (but valid) values.  The next
    # offset advances by the rows actually returned, so no row is hidden behind
    # a truncation boundary.
    while projected:
        out = {"schema_version": schema_version,
               "tasks": projected, "total": total, "offset": page_offset,
               "limit": page_limit,
               "next_offset": (page_offset + len(projected)
                                if page_offset + len(projected) < total else None)}
        encoded = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
        # The outer truncator counts characters. Use its actual authority, with
        # room for the tool envelope; complete JSON must survive that boundary.
        if len(encoded) <= max(1_000, int(result_limit) - 1_000):
            return out
        projected.pop()
    if page:
        raise ScheduleRefused(
            "projection_too_large",
            f"Schedule at offset {page_offset} exceeds the tool result limit. "
            "Read state/scheduled_tasks.json for its full identity; "
            f"the following row is at offset {page_offset + 1}.")
    return {"schema_version": schema_version, "tasks": [],
            "total": total, "offset": page_offset, "limit": page_limit,
            "next_offset": page_offset if page_offset < total else None}


def schedule_activity_projection(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full owner/UI schedule view with truthful lifecycle labels.

    The HTTP Activity surface needs its existing rows and does not share the
    model tool's result cap.  ``schedule_tool_projection`` is the bounded page
    used by ``manage_schedules``.
    """
    out = dict(data or {})
    tasks = []
    for raw in out.get("tasks") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        status = schedule_lifecycle_status(row)
        row["status"] = status
        row["active"] = status == "active"
        row["consumed"] = status == "consumed"
        row["suppressed"] = status == "suppressed"
        row["retained"] = status in {"consumed", "suppressed"}
        row["restorable"] = status == "suppressed"
        tasks.append(row)
    out["tasks"] = tasks
    return out


def _write_scheduled_tasks(data: Dict[str, Any], drive_root: pathlib.Path | None = None) -> None:
    # Author the stamp at the write seam (CPL4-C7): reads default it, but a
    # document that only ever gains its version in memory leaves the durable
    # file unversioned for every out-of-process reader.
    data.setdefault("schema_version", 1)
    path = _scheduled_tasks_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data, trailing_newline=True)


def _audit_row(record: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {key: record[key] for key in _AUDIT_ROW_KEYS if key in record}


def _audit_schedule_mutation(*, drive_root: pathlib.Path, operation_id: str, phase: str,
                             actor: str, task_id: str, action: str, schedule_id: str,
                             reason: str, result: str, before: Dict[str, Any] | None = None,
                             after: Dict[str, Any] | None = None) -> bool:
    """Append one bounded schedule audit fact to the EXISTING events log.

    Two facts share an ``operation_id``: the INTENT, written before anything is
    mutated, and the OUTCOME. A failed intent write stops the mutation, so the
    log cannot end up behind a change it never announced; a failed outcome write
    is disclosed to the caller instead of rolled back, because the change is
    already durable and an automatic undo would be a second unaudited mutation.
    """
    event = {
        "ts": utc_now_iso(), "type": "schedule_mutation", "phase": str(phase),
        "operation_id": str(operation_id), "actor": str(actor or "unknown"),
        "task_id": str(task_id or ""), "action": str(action),
        "id": str(schedule_id), "schedule_id": str(schedule_id),
        "reason": str(reason), "result": str(result),
    }
    if before is not None:
        event["before"] = _audit_row(before)
    if after is not None:
        event["after"] = _audit_row(after)
    try:
        from ouroboros.utils import append_jsonl

        return bool(append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", event))
    except Exception:
        log.exception("schedule audit %s write failed for %s", phase, schedule_id)
        return False


def sync_skill_schedules(skills: List[Any], *, drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Sync reviewed skill manifest scheduled_tasks into the core schedule table."""
    with schedule_transaction(drive_root):
        data = load_schedule_store(drive_root)
        by_id = {str(item.get("id") or ""): dict(item) for item in data.get("tasks") or []}
        touched: list[str] = []
        blocked_skill_names = {
            str(getattr(skill, "name", "") or "") for skill in skills
            if bool(getattr(skill, "identity_collision", False))
        }
        changed = False
        for skill in skills:
            if bool(getattr(skill, "identity_collision", False)):
                # Preserve prior rows: a collision is not a removed/runnable skill.
                continue
            manifest = getattr(skill, "manifest", None)
            for spec in list(getattr(manifest, "scheduled_tasks", []) or []):
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name") or "").strip()
                cron = str(spec.get("cron") or "").strip()
                if not name or not cron:
                    continue
                schedule_id = schedule_slug("skill", str(getattr(skill, "name", "")), name)
                touched.append(schedule_id)
                # Schedule readiness plus the supervised_task permission.
                try:
                    from ouroboros.skill_readiness import skill_readiness_for_execution
                    schedule_ready = skill_readiness_for_execution(pathlib.Path(drive_root or _queue().DRIVE_ROOT), skill).ready
                except Exception:
                    log.debug("skill schedule readiness probe failed for %s", getattr(skill, "name", ""), exc_info=True)
                    schedule_ready = False
                schedule_ready = schedule_ready and "supervised_task" in set(
                    getattr(manifest, "permissions", []) or []
                )
                record = by_id.get(schedule_id, {})
                trigger = {"type": "cron", "expr": cron}
                timing_changed = (
                    dict(record.get("trigger") or {}) != trigger
                    or str(record.get("timezone") or "") != str(spec.get("timezone") or "")
                )
                # The suppression marker is keyed by the SCHEDULE ID, so an owner
                # decision survives a manifest edit that changes the row's content
                # under the same id — the resync reconciles what the row says, not
                # whether the owner still wants it to fire.
                manual_override = str(record.get("manual_override") or "").strip().lower()
                next_record = {
                    **record,
                    "id": schedule_id,
                    "name": f"{getattr(skill, 'name', '')}/{name}",
                    "description": str(spec.get("description") or f"Scheduled skill task {getattr(skill, 'name', '')}/{name}"),
                    "enabled": False if manual_override in SUPPRESSED_OVERRIDES else bool(schedule_ready),
                    "timezone": str(spec.get("timezone") or ""),
                    "trigger": trigger,
                    "task": {
                        "type": "task",
                        "text": (
                            f"Run reviewed scheduled skill task `{getattr(skill, 'name', '')}/{name}`. "
                            "Use skill_exec or the reviewed extension surface as appropriate, then report outcome."
                        ),
                        "metadata": {
                            "source": "skill_scheduled_task",
                            "skill": str(getattr(skill, "name", "")),
                            "scheduled_task": name,
                        },
                    },
                    "source": "skill_manifest",
                    "skill": str(getattr(skill, "name", "")),
                    "skill_content_hash": str(getattr(skill, "content_hash", "")),
                    "updated_at": utc_now_iso(),
                }
                if timing_changed or not next_record.get("next_run_at"):
                    next_record["next_run_at"] = _schedule_next_run(next_record)
                if next_record != record:
                    by_id[schedule_id] = next_record
                    changed = True
        for schedule_id, record in list(by_id.items()):
            if (
                str(record.get("source") or "") == "skill_manifest"
                and str(record.get("skill") or "") not in blocked_skill_names
                and schedule_id not in touched
                and not _is_suppressed(record)
            ):
                by_id.pop(schedule_id, None)
                changed = True
        if changed:
            data["tasks"] = list(by_id.values())
            _write_scheduled_tasks(data, drive_root)
        return {"changed": changed, "skill_schedule_ids": touched}


def resync_skill_schedules(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Mirror discovered manifest schedules after skill lifecycle changes."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import discover_skills

    root = pathlib.Path(drive_root or _queue().DRIVE_ROOT)
    return sync_skill_schedules(
        discover_skills(root, repo_path=get_skills_repo_path()),
        drive_root=root,
    )


def _rows_hold_schedule(rows: Any, schedule_id: str) -> bool:
    """Whether any queue row (a PENDING task or a RUNNING/snapshot wrapper) is its task."""
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        task = row.get("task") if isinstance(row.get("task"), dict) else row
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if str(meta.get("schedule_id") or "") == schedule_id:
            return True
    return False


def _schedule_running_or_queued(schedule_id: str, drive_root: pathlib.Path | None = None) -> bool | None:
    """Whether a task this schedule already admitted is still pending or running.

    ``None`` means UNKNOWN, and it is load-bearing: PENDING/RUNNING are the
    SUPERVISOR process's live dicts, and ``manage_schedules`` runs in a worker
    process where those dicts are this process's own empty copies. Answering
    ``False`` from there would claim nothing is in flight — exactly the claim a
    lifecycle action must not make, since it governs future dispatch only.

    A worker therefore reads the durable queue snapshot instead. That file is
    rewritten on the transitions that change this answer (a scheduled admission,
    a dispatch, a task finishing), so it is the authoritative out-of-process
    record; when it is absent or unreadable the answer is unknown, not false.
    """
    if not schedule_id:
        return False
    if not in_worker_process():
        return (_rows_hold_schedule(_queue().PENDING, schedule_id)
                or _rows_hold_schedule(_queue().RUNNING.values(), schedule_id))
    snapshot = read_json_dict(pathlib.Path(drive_root or _queue().DRIVE_ROOT)
                              / "state" / "queue_snapshot.json")
    if not isinstance(snapshot, dict):
        return None
    # ``read_json_dict`` intentionally returns ``{}``/None for malformed or
    # absent state.  Missing rows are not evidence of an empty queue.  Require
    # both arrays and a recent supervisor timestamp before answering False.
    pending = snapshot.get("pending")
    running = snapshot.get("running")
    if not isinstance(pending, list) or not isinstance(running, list):
        return None
    for row in (*pending, *running):
        if not isinstance(row, dict):
            return None
        # Snapshot rows are either direct task dicts (legacy fixtures) or the
        # supervisor wrapper with a ``task`` object.  A present but malformed
        # wrapper cannot prove that this schedule is absent from flight.
        if "task" in row and not isinstance(row.get("task"), dict):
            return None
        task = row.get("task") if isinstance(row.get("task"), dict) else row
        if (
            "metadata" in task
            and task.get("metadata") is not None
            and not isinstance(task.get("metadata"), dict)
        ):
            return None
    stamp = snapshot.get("ts")
    try:
        parsed = datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        # A slightly future stamp can come from clock skew between supervisor
        # and worker; it is still a fresh write.  Only an old stamp is evidence
        # that this snapshot can no longer prove the absence of a live run.
        age = time.time() - parsed.timestamp()
        if age > _QUEUE_SNAPSHOT_MAX_AGE_SEC:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return (_rows_hold_schedule(snapshot.get("pending"), schedule_id)
            or _rows_hold_schedule(snapshot.get("running"), schedule_id))


def _task_from_schedule(record: Dict[str, Any]) -> Dict[str, Any]:
    template = dict(record.get("task") or {})
    owner_chat_id = _queue().load_state().get("owner_chat_id") or 0
    task_id = uuid.uuid4().hex[:8]
    session_id = str(template.get("session_id") or f"schedule-{record.get('id') or task_id}")
    raw_metadata = template.get("metadata") if isinstance(template.get("metadata"), dict) else {}
    metadata = {
        key: value for key, value in dict(raw_metadata).items()
        if key not in RESERVED_TEMPLATE_FIELDS
    }
    task = {
        "id": task_id,
        "type": "task",
        "text": str(template.get("text") or template.get("description") or record.get("description") or record.get("name") or "Scheduled task"),
        "description": str(template.get("description") or template.get("text") or record.get("description") or record.get("name") or "Scheduled task"),
        "chat_id": template.get("chat_id") if template.get("chat_id") not in (None, "") else owner_chat_id,
        "priority": int(template["priority"]) if str(template.get("priority") or "").strip().lstrip("-").isdigit() else None,
        "root_task_id": task_id,
        "session_id": session_id,
        "actor_id": "scheduler",
        "delegation_role": "root",
        "metadata": metadata,
    }
    for key in ("attachments", "context", "expected_output", "constraints", "deadline_at", "project_id"):
        if key in template:
            task[key] = template[key]
    allowed_resources = normalize_allowed_resources(template.get("allowed_resources") or metadata.get("allowed_resources") or {})
    if allowed_resources:
        task["allowed_resources"] = allowed_resources
    existing_contract = template.get("task_contract") if isinstance(template.get("task_contract"), dict) else {}
    if existing_contract:
        task["task_contract"] = existing_contract
    task["task_contract"] = build_task_contract(apply_consciousness_authority(task))
    presence = metadata.get("presence")
    workspace = task["task_contract"]["workspace"]
    if isinstance(presence, dict) and presence and workspace["root"]:
        task.update(
            workspace_root=workspace["root"], workspace_mode=workspace["mode"],
            memory_mode="shared",
        )
    task["metadata"]["schedule_id"] = str(record.get("id") or "")
    task["metadata"]["schedule_name"] = str(record.get("name") or "")
    task["metadata"]["schedule_trigger"] = dict(record.get("trigger") or {})
    task["metadata"]["task_contract"] = task["task_contract"]
    if allowed_resources:
        task["metadata"]["allowed_resources"] = allowed_resources
    if task.get("deadline_at"):
        task["metadata"]["deadline_at"] = task.get("deadline_at")
    task["metadata"].setdefault("source", "scheduled_task")
    return task


def _notification_chat_id(record: Dict[str, Any]) -> int:
    """The row's own positive chat, else the owner's chat, else Main.

    Never ``notification_chat_route``: chat 0 is a real destination there (the
    Skill Review panel) and a notification addressed to it reaches nobody.
    """
    try:
        pinned = int(record.get("chat_id") or 0)
    except (TypeError, ValueError):
        pinned = 0
    if pinned > 0:
        return pinned
    try:
        owner = int(_queue().load_state().get("owner_chat_id") or 0)
    except (TypeError, ValueError):
        owner = 0
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    return owner if owner > 0 else WEB_UI_CHAT_ID


def _notify_source_silenced(source: str) -> bool:
    """A skill's standing reminders fall silent with the skill: the resync never
    touches ``skill:`` rows, so a disabled or removed skill would otherwise keep
    ringing while its card says off."""
    if not source.startswith("skill:"):
        return False
    from ouroboros.skill_loader import find_skill, load_enabled

    name = source[len("skill:"):]
    return find_skill(_queue().DRIVE_ROOT, name) is None or not load_enabled(_queue().DRIVE_ROOT, name)


def _fire_owner_notification(record: Dict[str, Any], now: datetime.datetime,
                             scheduled_for: datetime.datetime) -> Dict[str, Any] | None:
    """Persist one due ``kind: "notify"`` row's notification; ``None`` = not fired.

    The durable append happens here, under the table lock the tick already
    holds (a local file write, like a task row's result); the topic publish is
    the caller's, after the lock. The frame key carries the due instant, so a
    recurring or re-armed reminder rings on every occurrence while a crash
    replay of the same occurrence still collapses on the client.
    """
    from ouroboros.event_bus import emit_owner_notification

    notification = record.get("notification") if isinstance(record.get("notification"), dict) else {}
    source = str(record.get("source") or "")
    # UTC, so the occurrence key does not depend on the host's zone setting.
    due_iso = scheduled_for.astimezone(datetime.timezone.utc).isoformat()
    try:
        return emit_owner_notification(
            _queue().DRIVE_ROOT, chat_id=_notification_chat_id(record), category="notice",
            text=str(notification.get("text") or ""), source=source or "scheduler",
            key=f"{str(notification.get('key') or '') or str(record.get('id') or '')}@{due_iso}",
            scheduled_for=due_iso, publish=False,
        )
    except ValueError as exc:
        _record_last_error(record, f"invalid notification: {exc}")
        return None


def check_scheduled_tasks() -> None:
    """Dispatch due schedules: a task row enqueues an ordinary root task, a
    ``kind: "notify"`` row emits one owner notification without a model turn."""
    global _last_skill_schedule_sync
    fired: List[Dict[str, Any]] = []
    with schedule_transaction(_queue().DRIVE_ROOT):
        now_monotonic = time.monotonic()
        if now_monotonic - _last_skill_schedule_sync >= _SKILL_SCHEDULE_SYNC_INTERVAL_SEC:
            _last_skill_schedule_sync = now_monotonic
            try:
                resync_skill_schedules(_queue().DRIVE_ROOT)
            except Exception:
                log.debug("Failed to sync skill schedules during scheduler tick", exc_info=True)
        try:
            data = load_schedule_store(_queue().DRIVE_ROOT)
        except ScheduleStoreUnreadable:
            # The pass writes the table back at its end; on unreadable bytes that
            # would replace every row with an empty document. Skip instead.
            log.error("Scheduled task store is unreadable; skipping this scheduler pass")
            return
        changed = False
        collision_names = None
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        for record in list(data.get("tasks") or []):
            if not isinstance(record, dict) or not record.get("enabled", True):
                continue
            schedule_id = str(record.get("id") or "").strip()
            if not schedule_id:
                record["id"] = uuid.uuid4().hex[:8]
                schedule_id = str(record["id"])
                changed = True
            trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
            trigger_type = str(trigger.get("type") or "cron").strip().lower()
            notify_row = str(record.get("kind") or "") == SCHEDULE_KIND_NOTIFY
            # A notify row admits no task, so nothing of it can be in flight.
            if not notify_row and _schedule_running_or_queued(schedule_id, _queue().DRIVE_ROOT) is not False:
                # Unknown reads as "still in flight": re-dispatching a schedule
                # whose previous run may be alive is worse than waiting a pass.
                continue
            tz = _timezone_for_schedule(record)
            now = now_utc.astimezone(tz)
            expr = ""
            if trigger_type == "once":
                # One-shot (B2b W=A): fires once at/after run_at via the same admission path
                # as cron, then is marked done below. A consumed receipt (non-empty completed_at)
                # NEVER re-fires even re-enabled from UI; re-arm = gateway upsert, fresh run_at.
                if record.get("completed_at"):
                    continue
                due, once_error = _once_due(trigger, tz, now)
                if once_error:
                    changed = _record_last_error(record, once_error) or changed
                    continue
                if not due:
                    continue
            elif trigger_type != "cron":
                changed = _record_last_error(record, f"unsupported trigger type: {trigger_type}") or changed
                continue
            else:
                expr = str(trigger.get("expr") or record.get("cron") or "").strip()
                if not expr:
                    changed = _record_last_error(record, "missing cron expression") or changed
                    continue
                next_run = _parse_schedule_time(record.get("next_run_at"), tz)
                if next_run is None:
                    try:
                        next_run = _next_cron_time(expr, now - datetime.timedelta(minutes=1))
                        record["next_run_at"] = next_run.isoformat()
                        changed = True
                    except Exception as exc:
                        changed = _record_last_error(record, f"{type(exc).__name__}: {exc}") or changed
                        continue
                if next_run > now:
                    continue
            if str(record.get("source") or "") == "skill_manifest":
                if collision_names is None:
                    collision_names = skill_identity_collision_names(_queue().DRIVE_ROOT)
                if str(record.get("skill") or "") in collision_names:
                    continue
            if notify_row:
                if _notify_source_silenced(str(record.get("source") or "")):
                    continue
                due_at = _parse_schedule_time(trigger.get("run_at"), tz) if trigger_type == "once" else next_run
                row = _fire_owner_notification(record, now, due_at or now)
                if row is None:
                    # Not fired: the row stays armed and its last_error says why
                    # (set by the helper for a rejected notification; an append
                    # that failed is retried on the next pass).
                    changed = _record_last_error(record, str(record.get("last_error") or "notification log write failed; retrying")) or changed
                    continue
                fired.append(row)
                record["last_run_at"] = now.isoformat()
                record["last_error"] = ""
                if trigger_type == "once":
                    record["enabled"] = False
                    record["completed_at"] = now.isoformat()
                    record["next_run_at"] = ""
                else:
                    try:
                        record["next_run_at"] = _next_cron_time(expr, now).isoformat()
                    except Exception as exc:
                        record["last_error"] = f"{type(exc).__name__}: {exc}"
                changed = True
                continue
            task = _task_from_schedule(record)
            try:
                from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

                write_task_result(
                    _queue().DRIVE_ROOT,
                    str(task["id"]),
                    STATUS_SCHEDULED,
                    root_task_id=str(task["id"]),
                    actor_id="scheduler",
                    delegation_role="root",
                    description=str(task.get("description") or task.get("text") or ""),
                    expected_output=str(task.get("expected_output") or ""),
                    constraints=str(task.get("constraints") or ""),
                    context=str(task.get("context") or ""),
                    allowed_resources=task.get("allowed_resources") if isinstance(task.get("allowed_resources"), dict) else {},
                    deadline_at=str(task.get("deadline_at") or ""),
                    task_contract=task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
                    result="Scheduled task queued.",
                    metadata=dict(task.get("metadata") or {}),
                    schedule_id=schedule_id,
                    schedule_name=str(record.get("name") or ""),
                )
            except Exception:
                log.debug("Failed to persist scheduled task result before enqueue", exc_info=True)
            admitted = _queue().enqueue_task(task)
            record["last_run_at"] = now.isoformat()
            record["last_task_id"] = task["id"]
            record_scheduled_admission(task, admitted, record)
            if trigger_type == "once":
                refused = isinstance(admitted, dict) and admitted.get("_admission_blocked")
                permanent = (refused == "project_routing_fence"
                             and admitted.get("_project_lifecycle") == "tombstoned")
                if not refused or permanent:
                    # A consumed receipt includes a permanent target refusal;
                    # keep its failed task and last_error. Transient refusals retry.
                    record["enabled"] = False
                    record["completed_at"] = now.isoformat()
                    record["next_run_at"] = ""
                elif str(refused).startswith("consciousness_"):
                    # The consciousness door refused (the tree's allowance or concurrency —
                    # a refusal that can last hours): the one-shot stays armed but its run
                    # point moves forward by the alarm floor, so it is not re-fired on every
                    # supervisor pass (a fresh task id, a failed result row and a ledger
                    # read per pass). The same class the evolution scheduler pauses on.
                    from ouroboros.config import get_bg_wakeup_min_sec

                    trigger["run_at"] = (now + datetime.timedelta(seconds=int(get_bg_wakeup_min_sec()))).isoformat()
                    record["trigger"] = trigger
            else:
                try:
                    record["next_run_at"] = _next_cron_time(expr, now).isoformat()
                except Exception as exc:
                    record["last_error"] = f"{type(exc).__name__}: {exc}"
            changed = True
        # Consumed one-shot receipts age out past the unified GC retention (DEVELOPMENT
        # Runtime Cleanup SSOT; enabled records are never pruned — see the helper).
        from ouroboros.retention import age_cutoff, get_gc_retention_days

        kept, pruned = _prune_consumed_once(list(data.get("tasks") or []),
                                            age_cutoff(get_gc_retention_days()))
        if pruned:
            data["tasks"], changed = kept, True
        if changed:
            _write_scheduled_tasks(data)
            _queue().persist_queue_snapshot(reason="scheduled_tasks")
    if fired:
        # Subscribers (the Telegram skill) run outside the table lock: their
        # handlers are not this tick's business, and the rows are already durable.
        from ouroboros.event_bus import publish_owner_notification

        for row in fired:
            publish_owner_notification(row)
