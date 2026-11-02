"""Durable schedules, their projections and due occurrence dispatch.

Owns state/scheduled_tasks.json and turns a due schedule into a queued task,
skipping any whose previous run is still pending or running.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import datetime
import json
import logging
import os
import pathlib
import threading
import uuid
from typing import Any, Callable, Dict, List
from ouroboros.consciousness_authority import apply_consciousness_authority
from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_resources
from ouroboros.dialogue_provenance import presence_metadata_binding
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS
from ouroboros.utils import atomic_write_json, in_worker_process, read_json_dict, utc_now_iso
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from supervisor.schedule_time import (
    next_cron_time as _next_cron_time,
    once_due as _once_due,
    parse_schedule_time as _parse_schedule_time,
    prune_consumed_once_records as _prune_consumed_once,
    record_last_error as _record_last_error,
    schedule_next_run as _schedule_next_run,  # noqa: F401 -- lifecycle writer seam
    timezone_for_schedule as _timezone_for_schedule,
)
from supervisor.schedule_view import _is_consumed_once, decorate, schedule_lifecycle_status  # noqa: F401 -- shared lifecycle words

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


# Per-thread depth of the held transactions, keyed by the TABLE's resolved lock
# path. Thread-local, so it needs no mutex of its own; the cross-thread ordering
# is the queue lock the transaction takes first.
_SCHEDULE_TX_STATE = threading.local()
_AUTOMATIC = contextvars.ContextVar("schedule_automatic_acquisitions", default=False)
_SCHEDULE_LOCK_SUFFIX = ".lock"
_SCHEDULE_LOCK_TIMEOUT_SEC = 8.0
_SCHEDULE_LOCK_STALE_SEC = 30.0
# The owner-governed actions over an EXISTING row; creation and editing stay on
# the upsert/patch seams, so this set is exactly "what happens to a future dispatch".
SCHEDULE_ACTIONS: frozenset[str] = frozenset({"disable", "delete", "restore"})
# Fields the RUNTIME owns. A caller builds its record from a GET that may already
# be stale, so these are always taken from the row on disk instead of the payload.
_RUNTIME_OWNED_FIELDS: tuple[str, ...] = (
    "source", "skill", "created_at", "last_run_at", "last_task_id", "last_error",
    "last_outcome", "previous_run_open",
    # The occurrence protocol's host facts (supervisor/schedule_occurrence.py).
    "occurrence", "hold", "continuation_of", "delete_requested_at", "delete_requested_by", "delete_reason",
)
# What an audit event may say about a row: lifecycle facts only. The task template
# is a private objective, never audit material, and would also be unbounded.
_AUDIT_ROW_KEYS: tuple[str, ...] = (
    "id", "name", "enabled", "source", "skill", "trigger", "timezone",
    "created_at", "updated_at", "last_run_at", "last_task_id", "completed_at",
    "next_run_at", "delete_requested_at", "delete_requested_by", "delete_reason", "writer_kind",
    "followup_relation", "followup_origin", "followup_hold", "followup_wait",
)
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
    whose contract is "the state is unknown, the write was refused". ``reason``
    keeps the acquisition outcome: only ``contention`` is a temporary gap.
    """

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = str(reason)


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
def automatic_acquisitions():
    """An automatic caller's Q-held schedule work: the off-loop scan/admission and the
    worker launch's final start. Under it every ``schedule_transaction`` tries the table
    once and runs its body under ``store.acquisition_budget(0)``, and ``control_guard``
    tries the origin's cancellation and launch locks once: confirmed contention defers
    that caller instead of every queue reader. Preparation between transactions keeps waits."""
    token = _AUTOMATIC.set(True)
    try:
        yield
    finally:
        _AUTOMATIC.reset(token)


def automatic() -> bool:
    return _AUTOMATIC.get()


@contextlib.contextmanager
def schedule_transaction(drive_root: pathlib.Path | None = None):
    """Serialize a whole schedule read/modify/write across threads and processes.

    The transaction takes BOTH locks ITSELF — the queue lock first, the table's
    sidecar file lock second — so no caller can compose the pair in the other
    order. Asking callers to order them was the bug: ``schedule_followup`` wraps
    its pending read and its write in one transaction, then ``upsert_scheduled_task``
    reached for the queue lock INSIDE that hold, against the scheduler tick's
    queue-then-table order. Owning the order here is what makes every caller
    provably consistent instead of individually audited.

    Reentrancy is keyed by the TABLE, not by a bare depth counter: a nested
    transaction on the same file rides the held file lock, while one addressing a
    different drive root still takes its own — a depth-only guard would let the
    second root's write run unserialized behind the first root's lock.
    """
    from ouroboros.store import acquisition_budget

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
        outcome: Dict[str, Any] = {}
        timeout = 0.0 if automatic() else _SCHEDULE_LOCK_TIMEOUT_SEC
        fd = acquire_exclusive_file_lock(
            lock_path, timeout_sec=timeout, stale_sec=_SCHEDULE_LOCK_STALE_SEC,
            owner_aware_stale=True, outcome=outcome)
        if fd is None:
            raise ScheduleLockTimeout(f"Could not acquire schedule lock {lock_path} within {timeout:g}s",
                                      reason=outcome.get("reason", "unknown"))
        held[key] = 1
        try:
            with acquisition_budget(0) if automatic() else contextlib.nullcontext():
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
    from ouroboros.installation_stop import confirm_absent

    path = _scheduled_tasks_path(drive_root)
    try:
        # lstat, not exists(): a dangling symlink at the table's path IS present.
        try:
            path.lstat()
        except FileNotFoundError:
            confirm_absent(path)
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
    # ``last_error`` says why a row did not do its job (a note whose chat write
    # was not confirmed reads "consumed" otherwise); it is bounded like any text.
    projection_keys = (
        "id", "name", "enabled", "source", "skill", "trigger",
        "created_at", "last_run_at", "last_task_id", "last_error", "completed_at", "next_run_at",
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

    template_key = "notification" if raw.get("kind") in {"notify", "note"} else "task"
    template = raw.get(template_key) if isinstance(raw.get(template_key), dict) else {}
    objective_source = ""
    for candidate in (template.get("objective"), template.get("text"),
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
    row["retained"] = status in {"consumed", "held", "delete_pending"}
    row["restorable"] = (status == "delete_pending" and not _is_consumed_once(raw)) or bool(
        raw.get("followup_hold") and raw.get("hold_persisted", True))
    row["delete_pending"] = status == "delete_pending"
    if row["delete_pending"]:
        row["delete_requested_at"] = _bounded_projection_text(raw.get("delete_requested_at"), 64)
    from supervisor.followup_policy import relation_kind, origin_of
    row["relation"] = relation_kind(raw)
    row["followup_origin"] = origin_of(raw)
    row["followup_hold"] = raw.get("followup_hold")
    row["followup_wait"] = raw.get("followup_wait", "")
    row["hold_persisted"] = raw.get("hold_persisted", bool(raw.get("followup_hold")))
    relation = raw.get("followup_relation") or {}
    row["billing_group"] = relation.get("billing_group")
    row["deadline_at"] = relation.get("deadline_at", "")
    # A due occurrence that WAITS (#1315): the typed reason is the fact; what to say
    # about it, if anything, is the mind's call.
    hold = raw.get("hold") if isinstance(raw.get("hold"), dict) else None
    if hold:
        row["waiting"] = {key: _bounded_projection_text(hold.get(key), 200) for key in ("reason", "detail", "since")}
    occurrence = raw.get("occurrence") if isinstance(raw.get("occurrence"), dict) else None
    if occurrence:
        row["occurrence"] = {key: str(occurrence.get(key) or "") for key in ("phase", "task_id", "due_at")}
    facts = decorate(raw, datetime.datetime.now(datetime.timezone.utc))
    for key in ("display_name", "kind", "timezone", "destination", "outcome", "previous_run_open", "waiting_detail", "withdrawal"):
        if key in facts:
            row[key] = facts[key]
    row["display_name"] = _bounded_projection_text(row["display_name"])
    return row


def presence_promise(binding: Any, row: Any) -> bool:
    """Whether ``row`` is a follow-up the exact nonempty Presence ``binding`` scheduled (#1589).

    The one ownership predicate behind a Presence caller's schedule reads, its context
    listing and its withdrawal: the binding carrier ``tools/followup.py`` copies by value
    onto the row's task template. An owner or skill row, a note, another binding and an
    empty or malformed carrier never match; objective text, room or writer kind decide nothing.
    """
    wanted = str(binding or "").strip()
    template = row.get("task") if isinstance(row, dict) else None
    return bool(wanted and isinstance(template, dict) and row.get("source") == "task_followup"
                and row.get("kind") not in {"notify", "note"}
                and presence_metadata_binding(template.get("metadata")) == wanted)


def presence_promise_store(data: Dict[str, Any], binding: Any) -> Dict[str, Any]:
    """The store narrowed to ``binding``'s own follow-ups, filtered BEFORE any page is cut."""
    return {**data, "tasks": [row for row in data.get("tasks") or [] if presence_promise(binding, row)]}


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
        row = decorate(raw, datetime.datetime.now(datetime.timezone.utc))
        status = schedule_lifecycle_status(row)
        row["status"] = status
        row["active"] = status == "active"
        row["consumed"] = status == "consumed"
        projected = _schedule_projection_row(row)
        for key in ("retained", "restorable", "relation", "followup_origin", "followup_hold",
                    "followup_wait", "hold_persisted", "billing_group", "deadline_at", "delete_pending"):
            row[key] = projected[key]
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
    if atomic_write_json(path, data, trailing_newline=True) is False:
        raise OSError("scheduled task store write returned False")


def _audit_row(record: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {key: record[key] for key in _AUDIT_ROW_KEYS if key in record}


def _audit_schedule_mutation(*, drive_root: pathlib.Path, operation_id: str, phase: str,
                             actor: str, task_id: str, action: str, schedule_id: str,
                             reason: str, result: str, before: Dict[str, Any] | None = None,
                             after: Dict[str, Any] | None = None,
                             changed_fields: list[str] | None = None) -> bool:
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
    if changed_fields is not None:
        event["changed_fields"] = [str(key)[:64] for key in changed_fields[:16]]
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


def _rows_hold_schedule(rows: Any, schedule_id: str, task_id: str = "") -> bool:
    """Whether any queue row (a PENDING task or a RUNNING/snapshot wrapper) is its task."""
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        task = row.get("task") if isinstance(row.get("task"), dict) else row
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if (schedule_id and str(meta.get("schedule_id") or "") == schedule_id) or (task_id and task.get("id") == task_id):
            return True
    return False


def _schedule_running_or_queued(schedule_id: str, drive_root: pathlib.Path | None = None, *, task_id: str = "") -> bool | None:
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
    if not (schedule_id or task_id):
        return False
    if not in_worker_process():
        return (_rows_hold_schedule(_queue().PENDING, schedule_id, task_id)
                or _rows_hold_schedule(_queue().RUNNING.values(), schedule_id, task_id))
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_status import queue_snapshot_observation
    snapshot = load_queue_document(drive_root or _queue().DRIVE_ROOT)
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
    observation = queue_snapshot_observation(snapshot)
    if observation["age_sec"] is None or observation["age_sec"] > _QUEUE_SNAPSHOT_MAX_AGE_SEC:
        return None
    return (_rows_hold_schedule(snapshot.get("pending"), schedule_id, task_id)
            or _rows_hold_schedule(snapshot.get("running"), schedule_id, task_id))


def _task_from_schedule(record: Dict[str, Any], *, task_id: str = "") -> Dict[str, Any]:
    from supervisor.followup_policy import normalize_template, bind_task
    from supervisor.schedule_notes import is_note
    if is_note(record):  # a note is shown as written, never re-read as a model task's objective
        raise ValueError("a kind=notify row is a note, not a task template")
    template = normalize_template(record)
    task_id = task_id or uuid.uuid4().hex[:8]
    # Membership, not truthiness: a template's explicit chat 0 is its hidden partition.
    has_chat = template.get("chat_id") not in (None, "")
    from supervisor.state import control_value

    known, owner_chat_id = (True, None) if has_chat else control_value(_queue().load_state(), "owner_chat_id")
    if not known:
        owner_chat_id = None
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
        "chat_id": template.get("chat_id") if has_chat else owner_chat_id,
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
    # The schedule's name is the run's authored name, as a caller's title is for any admitted run;
    # an unnamed schedule's run (the host's placeholder included) is named by its request's first line.
    from ouroboros.project_naming import admission_names
    from supervisor.schedule_view import authored_name
    title, coined = admission_names({"title": authored_name(record)}, task["description"])
    task.update({key: value for key, value in (("title", title), ("suggested_name", coined)) if value})
    from ouroboros.settings_scales import EFFORT_SCALE

    # The explicit start the upsert checked; a hand-edited unknown tier keeps the default.
    if template.get("reasoning_effort") in EFFORT_SCALE:
        task["reasoning_effort"] = template["reasoning_effort"]
    allowed_resources = normalize_allowed_resources(template.get("allowed_resources") or metadata.get("allowed_resources") or {})
    if allowed_resources:
        task["allowed_resources"] = allowed_resources
    existing_contract = template.get("task_contract") if isinstance(template.get("task_contract"), dict) else {}
    if existing_contract:
        task["task_contract"] = dict(existing_contract)
        if task.get("deadline_at"):
            task["task_contract"]["deadline_at"] = task["deadline_at"]
    bind_task(_queue().DRIVE_ROOT, task, record)
    from ouroboros.consciousness_authority import stamp_scheduled_work
    stamp_scheduled_work(task, record, raw_metadata)
    task["task_contract"] = build_task_contract(apply_consciousness_authority(task))
    workspace = task["task_contract"]["workspace"]
    if presence_metadata_binding(metadata) is not None and workspace["root"]:
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


def _scan_outcomes(root, snapshot):
    """Observe canonical settled results off-lock; missing/nonterminal reads retry.

    This is a cache of terminal-time facts. Once an id is observed, later result
    enrichment does not revise it; projections never open the task store.
    """
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_finalization import quiet_finish, terminal_host_notice_text
    from ouroboros.terminal_time import terminal_time_fact
    from ouroboros.store import acquisition_budget

    observed = {}
    for row in snapshot.get("tasks") or []:
        task_id = str(row.get("last_task_id") or "")
        if not task_id or (row.get("last_outcome") or {}).get("task_id") == task_id:
            continue
        try:
            with acquisition_budget(0):  # optional observation must not stall active rows
                result = load_task_result(root, task_id, strict=True)
            if not result or result.get("status") not in SETTLED_STATUSES:
                continue
            notice = terminal_host_notice_text(result)
            words = str(result.get("wake_summary") or "").strip()
            # A host notice is the run's public line, else the sentence it wrote for its human (whole),
            # else the end of a spoken answer; a quiet run's internal note never is.
            text = str(notice or ("" if words or result.get("wake_silent") is True else result.get("result")) or "").strip()
            observed[str(row.get("id") or "")] = {
                "task_id": task_id, "state": result["status"], "quiet": quiet_finish(result),
                "notice": bool(notice), "ended_at": terminal_time_fact(result)["occurred_at"],
                "line": text.splitlines()[-1][:160] if text else words,
            }
        except Exception:
            log.debug("schedule result %s not observed this pass", task_id, exc_info=True)
    return observed


def _remove_withdrawn(data, root, cutoff_epoch, *, reads=None):
    """Stage audited removals only after retention and settled work permit them.

    The caller records outcomes after its durable table write. Age is checked
    first: recent withdrawals never pay a receipt or settlement read for GC.
    """
    from supervisor.schedule_occurrence import owed, deletion_settled

    removals = []
    for record in list(data.get("tasks") or []):
        at = _parse_schedule_time(record.get("delete_requested_at"), datetime.timezone.utc)
        if at is None or at.timestamp() >= cutoff_epoch:
            continue
        if (isinstance(record.get("occurrence"), dict) or owed(record, drive_root=root, reads=reads) is not False
                or not deletion_settled(record, drive_root=root, reads=reads)):
            continue
        audit = dict(drive_root=root, operation_id=uuid.uuid4().hex[:12], actor="host", task_id="",
                     action="remove", schedule_id=str(record.get("id") or ""), reason="withdrawal_retention_elapsed")
        if not _audit_schedule_mutation(phase="intent", result="intended", before=record, **audit):
            continue
        data["tasks"].remove(record)
        removals.append((record, audit))
    return removals


def check_scheduled_tasks(*, stop: Callable[[], bool] | None = None) -> None:
    """Admit due cron/once schedules through the occurrence protocol.

    Claims and reconciliation run under the queue+table locks using signed
    receipt bodies prepared outside them, also for binding repair and retention.
    Resource preparation, root billing and the allowance run outside them;
    admission rechecks signatures/controls and commits under them again
    (``supervisor/schedule_occurrence.py``). A due ``kind: "notify"`` row is a
    note: consumed in this pass's one table write, shown after the locks are
    released (``supervisor/schedule_notes.py``), never a task.

    The supervisor runs it off the loop under ``automatic_acquisitions`` with its
    generation's ``stop`` (``server_maintenance``): confirmed contention under Q
    defers that row, or the pass, writing no hold, audit or claim, while
    unreadable, corrupt or foreign records keep their typed holds."""
    from supervisor import schedule_notes as notes, schedule_occurrence as occurrences

    if stop is not None and stop():
        return
    root = _queue().DRIVE_ROOT
    from supervisor.followup_policy import control_guard, scan_policy, refresh_policy

    # Copy the table while its writers are excluded. Policy reads below own no Q
    # or table lock; a changed row/restart is retried rather than overwritten.
    with schedule_transaction(root):
        try:
            snapshot = copy.deepcopy(load_schedule_store(root))
        except ScheduleStoreUnreadable:
            log.error("Scheduled task store is unreadable; skipping this scheduler pass")
            return
    outcomes = _scan_outcomes(root, snapshot)
    observed = scan_policy(root, snapshot)
    from ouroboros.retention import age_cutoff, get_gc_retention_days

    cutoff = age_cutoff(get_gc_retention_days())
    receipt_rows = [row for row, _view in observed.values()]
    for row in snapshot.get("tasks") or []:
        # Idle historical rows need no body except when retention can remove them.
        at = _parse_schedule_time(row.get("delete_requested_at") or (
            row.get("completed_at") if _is_consumed_once(row) else ""), datetime.timezone.utc)
        if at is not None and at.timestamp() < cutoff:
            receipt_rows.append(row)
    reads = occurrences.prepare_receipts(receipt_rows, drive_root=root)
    claims: List[Dict[str, Any]] = []
    due_notes: List[Dict[str, Any]] = []
    with schedule_transaction(root):
        if stop is not None and stop():
            return
        try:
            data = load_schedule_store(root)
        except ScheduleStoreUnreadable:
            # The pass writes the table back at its end; on unreadable bytes that
            # would replace every row with an empty document. Skip instead.
            log.error("Scheduled task store is unreadable; skipping this scheduler pass")
            return
        changed = False
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        for record in list(data.get("tasks") or []):
            if not isinstance(record, dict):
                continue
            unchanged = copy.deepcopy(record)
            outcome = outcomes.get(str(record.get("id") or ""))
            if outcome and record.get("last_task_id") == outcome["task_id"] and record.get("last_outcome") != outcome:
                record["last_outcome"] = outcome
                changed = True
            previous_open = record.get("previous_run_open")
            if previous_open and _schedule_running_or_queued("", root, task_id=previous_open["task_id"]) is False:
                record.pop("previous_run_open", None)
                changed = True
            from supervisor.schedule_lifecycle import _adopt_stored_row

            try:
                _adopt_stored_row(root, data, record)
            except ScheduleRefused:
                continue
            prior, view = observed.get(str(record.get("id") or ""), (None, None))
            if (unchanged != prior or data.get("followup_restart") != snapshot.get("followup_restart")):
                continue
            try:
                with control_guard(root, record):
                    occurrences.repair_followup_binding(record, reads=reads)
                    view = refresh_policy(root, data, record, observed=view)
            except Exception as exc:
                if not isinstance(exc, occurrences.ReceiptChanged) and not occurrences.contention(exc):
                    log.exception("Follow-up policy persistence unavailable")
                continue
            if view.get("busy"):
                continue
            if isinstance(record.get("occurrence"), dict):
                # An occurrence in flight is decided from durable facts first — even on a
                # disabled row: admission already happened, later changes are future-only.
                if occurrences.holding(record, now_utc):
                    continue
                try:
                    verdict, stored = occurrences.reconcile(record, reads=reads)
                except Exception as exc:
                    if occurrences.contention(exc):
                        continue
                    raise
                changed = changed or verdict not in {"live", "retry"}
                if verdict == "reconsider":
                    # Unstarted work follows the current authored timing/control.
                    occurrences.discard_claim(record)
                elif verdict == "settled":
                    occurrences.settle(record)
                elif verdict in {"prepare", "republish"}:
                    claims.append(occurrences.view(record, stored))
                    continue
                else:
                    continue
            if record.get("delete_requested_at") or not record.get("enabled", True) or occurrences.holding(record, now_utc):
                continue
            schedule_id = str(record.get("id") or "").strip()
            if not schedule_id:
                record["id"] = uuid.uuid4().hex[:8]
                schedule_id = str(record["id"])
                changed = True
            trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
            trigger_type = str(trigger.get("type") or "cron").strip().lower()
            tz = _timezone_for_schedule(record)
            now = now_utc.astimezone(tz)
            if trigger_type == "once":
                # One-shot (B2b W=A): fires once at/after run_at, then is consumed at
                # admission. A consumed receipt (non-empty completed_at) NEVER re-fires
                # even re-enabled from UI; re-arm = gateway upsert, fresh run_at.
                if record.get("completed_at"):
                    continue
                due, once_error = _once_due(trigger, tz, now)
                if once_error:
                    changed = _record_last_error(record, once_error) or changed
                    continue
                if not due:
                    continue
                due_at = str(trigger.get("run_at") or "")
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
                due_at = next_run.isoformat()
            running = _schedule_running_or_queued(schedule_id, root)
            if running is not False:
                previous = str(record.get("last_task_id") or "")
                if (running is True and previous and (record.get("previous_run_open") or {}).get("task_id") != previous
                        and _schedule_running_or_queued("", root, task_id=previous) is True):
                    record["previous_run_open"] = {"task_id": previous, "since": now_utc.isoformat()}
                    changed = True
                continue  # unknown queue state never licenses a duplicate firing
            if view.get("hold") or view["wait"] or record.get("followup_hold"):
                continue
            if notes.is_note(record):
                # Notes publish here, unlike task claims whose admission rechecks
                # controls later. Their final control fence must therefore be here.
                with control_guard(root, record):
                    view = refresh_policy(root, data, record)
                    if view.get("hold") or view["wait"] or record.get("followup_hold"):
                        continue
                    note, note_changed = notes.consume(record, due_at, now)
                    due_notes.extend([note] if note else [])
                    changed = changed or note_changed
                continue
            claims.append(occurrences.claim(record, due_at))
            changed = True
        # Consumed one-shot receipts age out past the unified GC retention (DEVELOPMENT
        # Runtime Cleanup SSOT; enabled records are never pruned — see the helper).
        removals = _remove_withdrawn(data, root, cutoff, reads=reads)
        changed = bool(removals) or changed
        kept, pruned = _prune_consumed_once(list(data.get("tasks") or []), cutoff,
            work_settled=lambda row: occurrences.deletion_settled(row, drive_root=root, reads=reads))
        if pruned:
            data["tasks"], changed = kept, True
        if changed:
            if _write_scheduled_tasks(data) is False:
                return
            for before, audit in removals:
                if not _audit_schedule_mutation(phase="outcome", result="removed", before=before, **audit):
                    log.error("schedule %s removed but its audit outcome is incomplete", before.get("id"))
            _queue().persist_queue_snapshot(reason="scheduled_tasks")
    notes.deliver(due_notes, root)
    occurrences.retain_deferred(data.get("tasks") or [])
    prepared = []
    for claimed in claims:
        if stop is not None and stop():
            return  # the claims stay durable: a later generation prepares the same identities
        prepared.append(occurrences.prepare(claimed))
    if prepared:
        occurrences.admit(prepared, stop=stop)
