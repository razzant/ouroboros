"""Canonical Project dialogue projections and routing annotations.

Project conversion stores a reference to the original owner row on the immutable
task binding. A Project room projects that row instead of copying it into
``chat.jsonl``. Terminal task projections append to that same canonical biography;
``chat_annotations.jsonl`` remains presentation-only and never owns routing or
Project state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
from ouroboros.utils import append_jsonl, iter_jsonl_objects, jsonl_append_lock_path, replace_atomic, strip_markdown, utc_now_iso

_ANNOTATIONS_NAME = "chat_annotations.jsonl"
_COMPACT_AT_BYTES = 800_000
_RETAINED_ARCHIVES = 3
log = logging.getLogger(__name__)


def _row_chat_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("chat_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _chat_paths(drive_root: Any) -> List[pathlib.Path]:
    root = pathlib.Path(drive_root)
    archives = sorted(
        (root / "archive").glob("chat_*.jsonl"),
        key=lambda path: path.name,
        reverse=True,
    )[:_RETAINED_ARCHIVES]
    return [*reversed(archives), root / "logs" / "chat.jsonl"]


def _text_sha256(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_owner_message_ref(
    *,
    chat_id: int,
    client_message_id: str,
    ts: str,
    text: str,
) -> Dict[str, Any]:
    """The canonical owner-row identity, built AT INGRESS from host-known facts.

    Identity is captured where the host writes the canonical row and passed by
    value downstream (BIBLE P2/P5); it is never re-derived from content later.
    ``text_sha256`` rides along as an integrity checksum, not a lookup key."""
    return {
        "chat_id": int(chat_id or 0),
        "client_message_id": str(client_message_id or ""),
        "ts": str(ts or ""),
        "text_sha256": _text_sha256(text),
    }


def owner_message_ref_is_valid(ref: Any) -> bool:
    """Whether a source ref has the complete host-minted owner-row identity."""
    if not isinstance(ref, dict) or not {
        "chat_id", "client_message_id", "ts", "text_sha256",
    }.issubset(ref):
        return False
    digest = ref.get("text_sha256")
    return bool(
        isinstance(ref.get("chat_id"), int)
        and not isinstance(ref.get("chat_id"), bool)
        and isinstance(ref.get("client_message_id"), str)
        and isinstance(ref.get("ts"), str) and ref.get("ts")
        and isinstance(digest, str) and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def source_refs_for_project(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Canonical owner-row references held by bindings for one Project lens."""
    from ouroboros.projects_registry import project_task_bindings

    refs: List[Dict[str, Any]] = []
    for row in project_task_bindings(drive_root).values():
        try:
            same_chat = int(row.get("project_chat_id") or 0) == int(project_chat_id or 0)
        except (TypeError, ValueError):
            same_chat = False
        ref = row.get("source_ref")
        if same_chat and isinstance(ref, dict) and ref:
            refs.append(dict(ref))
    return refs


def project_origin_rows(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Origin rows a Project lens can SYNTHESIZE when the canonical row is gone.

    Only bindings that carry ``source_text`` qualify (cross-thread origins — the
    binding is the retention-proof copy of the message that started the project).
    Deduplicated by complete origin identity so several bindings created from one
    owner message yield one row."""
    from ouroboros.projects_registry import project_task_bindings

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for row in project_task_bindings(drive_root).values():
        try:
            same_chat = int(row.get("project_chat_id") or 0) == int(project_chat_id or 0)
        except (TypeError, ValueError):
            same_chat = False
        ref = row.get("source_ref")
        text = row.get("source_text")
        if not (same_chat and isinstance(ref, dict) and ref and isinstance(text, str) and text):
            continue
        identity = (
            str(ref.get("chat_id") or ""),
            str(ref.get("client_message_id") or ""),
            str(ref.get("ts") or ""),
            str(ref.get("text_sha256") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"ref": dict(ref), "text": text})
    return rows


def project_recent_dialogue(
    memory: Any, project_chat_id: int, max_entries: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Focused recent rows plus retention-proof cross-thread owner origins."""
    from ouroboros.projects_registry import all_task_bindings

    try:
        bound = all_task_bindings(memory.drive_root)
    except Exception:
        bound = {}
    refs = source_refs_for_project(memory.drive_root, project_chat_id)
    ref_keys = {key for ref in refs if (key := _source_ref_identity(ref)) is not None}
    entries, coverage = memory.read_unconsolidated_chat(
        memory.load_dialogue_meta(), max_entries,
        predicate=lambda row: (
            _row_chat_id(row) == project_chat_id
            or bound.get(str(row.get("task_id") or "")) == project_chat_id
            or bool(_entry_source_identities(row) & ref_keys)
        ),
    )
    present_ref_keys = set()
    for entry in entries:
        present_ref_keys.update(_entry_source_identities(entry))
    retained: List[Dict[str, Any]] = []
    for origin in project_origin_rows(memory.drive_root, project_chat_id):
        ref = origin.get("ref") if isinstance(origin.get("ref"), dict) else {}
        if _source_ref_identity(ref) in present_ref_keys:
            continue
        retained.append({
            "chat_id": ref.get("chat_id"), "client_message_id": ref.get("client_message_id"),
            "ts": ref.get("ts"), "direction": "in", "text": origin.get("text"),
            "project_origin_projection": True,
        })
    return entries, coverage, retained


def _source_ref_identity(ref: Dict[str, Any]) -> Optional[tuple]:
    try:
        chat_id = int(ref.get("chat_id") or 0)
    except (TypeError, ValueError):
        return None
    return (
        chat_id, str(ref.get("client_message_id") or ""), str(ref.get("ts") or ""),
        str(ref.get("text_sha256") or ""),
    )


def _entry_source_identities(entry: Dict[str, Any]) -> set:
    if str(entry.get("direction") or "") != "in":
        return set()
    try:
        chat_id = int(entry.get("chat_id", 1) or 1)
    except (TypeError, ValueError):
        chat_id = 1
    client_id = str(entry.get("client_message_id") or "")
    ts = str(entry.get("ts") or "")
    text_hash = _text_sha256(entry.get("text"))
    return {
        (chat_id, client_id, ts, text_hash), (chat_id, "", ts, text_hash),
        (chat_id, client_id, "", text_hash), (chat_id, "", "", text_hash),
    }


def entry_matches_source_ref(entry: Dict[str, Any], refs: Iterable[Dict[str, Any]]) -> bool:
    """Whether ``entry`` is the original row identified by one binding ref."""
    ref_keys = {
        key for ref in refs if isinstance(ref, dict)
        if (key := _source_ref_identity(ref)) is not None
    }
    return bool(_entry_source_identities(entry) & ref_keys)


def resolve_owner_message_source(drive_root: Any, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Stream the exact named owner source across the durable generation chain."""
    from ouroboros.consolidator import _ordered_chat_generation_paths

    live = pathlib.Path(drive_root) / "logs" / "chat.jsonl"
    ref_key = _source_ref_identity(ref)
    if ref_key is None:
        return None
    for path in reversed(_ordered_chat_generation_paths(live)):
        try:
            for row in iter_jsonl_objects(path):
                if ref_key in _entry_source_identities(row):
                    return dict(row)
        except OSError:
            continue
    return None


def _latest_annotations(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl_objects(path):
        message_id = str(row.get("client_message_id") or "")
        if message_id and row.get("type") == "chat_annotation":
            latest[message_id] = dict(row)
    return latest


def latest_chat_annotations(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Latest presentation annotation per message; a torn tail is ignored."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    return _latest_annotations(path)


def chat_annotation_receipt(
    drive_root: Any, client_message_id: str, routing_token: str,
) -> Dict[str, Any]:
    """Return the exact token-bound annotation for one routing attempt."""
    row = latest_chat_annotations(drive_root).get(str(client_message_id or ""), {})
    if str(row.get("routing_token") or "") != str(routing_token or ""):
        return {}
    return dict(row)


def _compact_annotations_locked(drive_root: Any, path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size < _COMPACT_AT_BYTES:
        return
    retained_ids = {
        str(row.get("client_message_id") or "")
        for chat_path in _chat_paths(drive_root)
        for row in iter_jsonl_objects(chat_path)
        if row.get("client_message_id")
    }
    rows = [
        row for message_id, row in _latest_annotations(path).items()
        if message_id in retained_ids
    ]
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    replace_atomic(tmp, path)


def append_chat_annotation(
    drive_root: Any,
    client_message_id: str,
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    routing_token: str = "",
    reason: str = "",
    detail: str = "",
    options: Any = None,
    attachment_manifest: Any = None,
    require_latest_status: Any = None,
    require_latest_token: Any = None,
) -> bool:
    """Append one compact UI annotation.

    Presentation-first with ONE named exception (#198): a routing refusal row
    (status=needs_manual_target) is also the picker's durable decision-card
    authority — its token+options validate the owner's click, and the
    dispatch_pending/closing rows carry the click's first-wins/idempotency
    facts. Routing STATE still lives in the supervisor receipts (task-result
    admission, mailbox); the sidecar only arbitrates the card.

    ``require_latest_status`` (a set of status strings) turns the append into
    a compare-and-append under the annotations lock: the row is written only
    while the message's CURRENT latest status is in the set — the first-wins
    claim seam of the routing picker (#198). Absent/None keeps plain append.
    """
    message_id = str(client_message_id or "").strip()
    if not message_id:
        return False
    row = {
        "ts": utc_now_iso(),
        "type": "chat_annotation",
        "client_message_id": message_id[:200],
        "action": str(action or "")[:80],
        "target": str(target or "")[:200],
        "status": str(status or "")[:80],
    }
    if str(target_label or ""):
        row["target_label"] = str(target_label)[:200]
    if str(routing_token or ""):
        row["routing_token"] = str(routing_token)[:128]
    if str(reason or ""):
        row["reason"] = str(reason)[:200]
    if str(detail or ""):
        row["detail"] = str(detail)[:1000]
    if isinstance(options, list):
        row["options"] = [dict(item) for item in options[:100] if isinstance(item, dict)]
    if isinstance(attachment_manifest, list):
        row["attachment_manifest"] = [
            dict(item) for item in attachment_manifest if isinstance(item, dict)
        ]
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        return False
    try:
        if require_latest_status is not None or require_latest_token is not None:
            latest = _latest_annotations(path).get(message_id)
            if latest is not None:
                latest_status = str(latest.get("status") or "")
                latest_token = str(latest.get("routing_token") or "")
                if require_latest_status is not None and latest_status not in set(require_latest_status):
                    return False  # lost the claim race — the caller reads the truth back
                if require_latest_token is not None and latest_token not in set(require_latest_token):
                    return False  # a NEWER routing attempt owns the card now
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        _compact_annotations_locked(drive_root, path)
        return True
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def routing_target_label(
    drive_root: Any, action: str, target: str, *, task: Any = None,
    project_id: str = "",
) -> str:
    """Resolve one deterministic event-time label for an existing raw target."""
    target = str(target or "").strip()
    if not target:
        return ""
    try:
        from ouroboros.projects_registry import get_reserved_project, task_presentation_snapshot

        if action == "project_route":
            project = get_reserved_project(drive_root, target) or {}
            name = str(project.get("name") or "").strip()
            return name if name and name != target else "Project"
        return task_presentation_snapshot(
            drive_root, target, task=task, project_id=project_id,
        )["target_label"]
    except Exception:
        log.debug("Routing target label resolution failed for %s", target, exc_info=True)
        return "Task"


def routing_options_with_labels(drive_root: Any, options: Any) -> List[Dict[str, Any]]:
    """Stamp human labels on manual task choices while retaining their raw ids."""
    rows: List[Dict[str, Any]] = []
    for raw in list(options or [])[:100]:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        task_id = str(row.get("task_id") or "").strip()
        if task_id and not str(row.get("label") or "").strip():
            row["label"] = routing_target_label(
                drive_root, str(row.get("action") or "steer_task"), task_id,
                task=row, project_id=str(row.get("project_id") or ""),
            )
        rows.append(row)
    return rows


def routing_option_label(option: Any) -> str:
    """One human label per manual-routing option — the HOST SSOT (the durable
    routing_options history row and the Telegram skill both render through it;
    web mirrors it as chat_activity.routingOptionLabel)."""
    if not isinstance(option, dict):
        return ""
    if str(option.get("label") or "").strip():
        return str(option["label"]).strip()
    if str(option.get("action") or "") == "new_task_in_project":
        return f"New task in {str(option.get('project_name') or 'Project')}"
    if option.get("title") or option.get("project_name"):
        return str(option.get("title") or option.get("project_name"))
    return "Project" if option.get("project_id") and not option.get("task_id") else "Task"


def completion_status_label(result: Dict[str, Any], event: Dict[str, Any]) -> str:
    from ouroboros.task_results import (
        STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE,
    )

    status = str(result.get("status") or event.get("status") or "").strip().lower()
    axes = {}
    for source in (event, result):
        value = source.get("outcome_axes")
        if isinstance(value, dict):
            axes.update({key: axis for key, axis in value.items() if isinstance(axis, dict)})
    axis_status = {key: str(axis.get("status") or "").lower() for key, axis in axes.items()}
    failed = (
        status == STATUS_FAILED
        or axis_status.get("lifecycle") == STATUS_FAILED
        or axis_status.get("execution") in {"failed", "infra_failed"}
        or axis_status.get("objective") == "fail"
        or axis_status.get("review") == "fail"
        or axis_status.get("artifacts") in {"failed", "missing"}
        or str(result.get("artifact_status") or event.get("artifact_status") or "").lower()
        in {"failed", "missing"}
    )
    degraded = any(value in {"degraded", "partial", "best_effort"}
                   for value in axis_status.values())
    checkpoint = result.get("root_phase_checkpoint")
    degraded |= bool(isinstance(checkpoint, dict)
                     and str(checkpoint.get("post_task_synthesis") or "").lower() == "degraded")
    if status == STATUS_CANCELLED:
        return "Cancelled"
    if failed:
        return "Failed"
    if status == STATUS_COMPLETED:
        return "Completed with limitations" if degraded else "Completed"
    if status == STATUS_REJECTED_DUPLICATE:
        return "Not started"
    return status.replace("_", " ").title() or "Finished"


def append_canonical_task_summary(drive_root: Any, row: Dict[str, Any]) -> bool:
    """Append one task-summary row through the existing concurrent JSONL owner."""
    if not str(row.get("summary_id") or "").strip():
        return False
    path = pathlib.Path(drive_root) / "logs" / "chat.jsonl"
    return append_jsonl(path, dict(row))


def append_authored_task_summary(
    canonical_root: Any, result_root: Any, row: Dict[str, Any], *, status: str = "",
) -> bool:
    """Append the authored row and persist its identical continuation narrative."""
    appended = append_canonical_task_summary(canonical_root, row)
    persist_continuation_narrative(
        result_root,
        str(row.get("task_id") or ""),
        str(row.get("text") or ""),
        summary_id=str(row.get("summary_id") or ""),
        summary_kind=str(row.get("summary_kind") or ""),
        result_ref=row.get("result_ref") if isinstance(row.get("result_ref"), dict) else {},
        source_coverage=row.get("source_coverage") if isinstance(row.get("source_coverage"), dict) else {},
        status=status,
    )
    return appended


def _narrative_result_ref_is_valid(value: Any, task_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        str(value.get("kind") or "") == "task_result"
        and str(value.get("task_id") or "") == str(task_id or "")
        and str(value.get("reader") or "") == "get_task_result"
    )


def continuation_narrative_is_valid(value: Any, task_id: str) -> bool:
    """Validate the small, authored summary persisted beside a task result."""
    if not isinstance(value, dict) or not str(value.get("text") or "").strip():
        return False
    tid = str(task_id or "").strip()
    if not tid or str(value.get("task_id") or "") != tid:
        return False
    if str(value.get("summary_kind") or "") != "authored_root_summary":
        return False
    if str(value.get("summary_id") or "") != f"task-narrative:{tid}":
        return False
    result_ref = value.get("result_ref")
    coverage = value.get("source_coverage")
    return bool(
        _narrative_result_ref_is_valid(result_ref, tid)
        and isinstance(coverage, dict)
        and _narrative_result_ref_is_valid(coverage.get("task_result"), tid)
        and coverage.get("task_result") == result_ref
    )


def persist_continuation_narrative(
    drive_root: Any,
    task_id: str,
    text: str,
    *,
    summary_id: str,
    summary_kind: str,
    result_ref: Dict[str, Any],
    source_coverage: Dict[str, Any],
    status: str = "",
) -> bool:
    """Persist the exact authored summary through the task-result lock."""
    tid = str(task_id or "").strip()
    narrative = {
        "text": str(text or ""),
        "task_id": tid,
        "summary_id": str(summary_id or ""),
        "summary_kind": str(summary_kind or ""),
        "result_ref": dict(result_ref) if isinstance(result_ref, dict) else {},
        "source_coverage": dict(source_coverage) if isinstance(source_coverage, dict) else {},
        "written_at": utc_now_iso(),
    }
    if not tid or not continuation_narrative_is_valid(narrative, tid):
        return False
    try:
        from ouroboros.task_results import load_task_result, write_task_result

        existing = load_task_result(drive_root, tid) or {}
        if not existing and not str(status or "").strip():
            return False
        requested_status = str(status or existing.get("status") or "running")

        def _project(current: Dict[str, Any], _patch: Dict[str, Any]) -> Dict[str, Any]:
            current_narrative = current.get("continuation_narrative")
            if continuation_narrative_is_valid(current_narrative, tid):
                # The summary id is task-unique.  A second post-task worker must
                # not race a complete narrative with a partial/empty rewrite.
                return {
                    "status": str(current.get("status") or requested_status),
                    "continuation_narrative": dict(current_narrative),
                }
            return {
                "status": str(current.get("status") or requested_status),
                "continuation_narrative": dict(narrative),
            }

        write_task_result(
            drive_root, tid, requested_status, _field_projector=_project,
        )
        return True
    except Exception:
        log.warning("Failed to persist continuation narrative for %s", tid, exc_info=True)
        return False


def _bounded_chat_tail_rows(
    path: pathlib.Path, *, max_bytes: int, max_rows: int,
) -> List[Dict[str, Any]]:
    """Read only a bounded tail; never enter the unbounded archive resolver."""
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - max(1, int(max_bytes)))
            handle.seek(start)
            if start:
                handle.readline()  # discard the partial first JSONL row
            rows: List[Dict[str, Any]] = []
            for raw in handle:
                if len(rows) >= max(1, int(max_rows)):
                    break
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
            return rows
    except OSError:
        return []


def resolve_legacy_continuation_narrative(
    drive_root: Any, task_id: str, result_ref: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Find one recent authored row by exact task identity, within fixed bounds."""
    from ouroboros.context_budget import (
        CONTINUATION_NARRATIVE_LEGACY_GENERATIONS,
        CONTINUATION_NARRATIVE_LEGACY_MAX_ROWS,
        CONTINUATION_NARRATIVE_LEGACY_TAIL_BYTES,
    )

    tid = str(task_id or "").strip()
    expected_ref = dict(result_ref) if isinstance(result_ref, dict) else {}
    if not tid or not _narrative_result_ref_is_valid(expected_ref, tid):
        return None
    paths = _chat_paths(drive_root)
    paths = paths[-max(1, int(CONTINUATION_NARRATIVE_LEGACY_GENERATIONS)):]
    for path in reversed(paths):
        rows = _bounded_chat_tail_rows(
            path,
            max_bytes=CONTINUATION_NARRATIVE_LEGACY_TAIL_BYTES,
            max_rows=CONTINUATION_NARRATIVE_LEGACY_MAX_ROWS,
        )
        for row in reversed(rows):
            if (
                str(row.get("type") or "") != "task_summary"
                or str(row.get("summary_kind") or "") != "authored_root_summary"
                or str(row.get("summary_id") or "") != f"task-narrative:{tid}"
                or str(row.get("task_id") or "") != tid
                or not _narrative_result_ref_is_valid(row.get("result_ref"), tid)
                or row.get("result_ref") != expected_ref
                or not isinstance(row.get("source_coverage"), dict)
                or row["source_coverage"].get("task_result") != expected_ref
                or not str(row.get("text") or "").strip()
            ):
                continue
            return {
                "text": str(row.get("text") or ""),
                "task_id": tid,
                "summary_id": f"task-narrative:{tid}",
                "summary_kind": "authored_root_summary",
                "result_ref": dict(expected_ref),
                "source_coverage": dict(row["source_coverage"]),
                "source": {
                    "kind": "chat_jsonl",
                    "path": str(path),
                    "summary_id": str(row.get("summary_id") or ""),
                },
                "written_at": str(row.get("ts") or ""),
            }
    return None


def _append_terminal_task_projection(
    drive_root: Any, task_id: str, task: Dict[str, Any], result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> bool:
    """Project one terminal child result into canonical cognition, without an LLM."""
    from ouroboros.task_results import resolve_task_lineage, write_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    tid = str(task_id or "").strip()
    task = task if isinstance(task, dict) else {}
    result = result if isinstance(result, dict) else {}
    event = task_done_event if isinstance(task_done_event, dict) else {}
    if not tid or any(bool(row.get("_ephemeral") or row.get("ephemeral_decision"))
                      for row in (task, result, event)):
        return False
    lineage = resolve_task_lineage(
        tid,
        metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
        root_task_id=result.get("root_task_id") or task.get("root_task_id"),
        parent_task_id=result.get("parent_task_id") or task.get("parent_task_id"),
        delegation_role=result.get("delegation_role") or task.get("delegation_role"),
        original_task_id=result.get("original_task_id") or task.get("original_task_id"),
        timeout_retry_from=result.get("timeout_retry_from") or task.get("timeout_retry_from"),
    )
    status = str(result.get("status") or event.get("status") or "").strip().lower()
    if status not in SETTLED_STATUSES:
        return False
    is_root = bool(lineage["is_root_task"])
    summary_id = f"task-terminal:{tid}"
    summary_kind = "terminal_root_projection" if is_root else "terminal_result_projection"
    parent_id = str(lineage.get("parent_task_id") or "")
    root_id = str(lineage.get("root_task_id") or tid)
    from ouroboros.project_facts import resolve_project_id

    appended = False

    def _append_once(current: Dict[str, Any], _patch: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal appended
        existing_marker = current.get("canonical_terminal_projection")
        if isinstance(existing_marker, dict) and str(existing_marker.get("summary_id") or "") == summary_id:
            return {"status": str(current.get("status") or status)}
        checkpoint = current.get("root_phase_checkpoint")
        post_task_phase = (
            str(checkpoint.get("post_task_synthesis") or "")
            if isinstance(checkpoint, dict) else ""
        )
        if is_root and post_task_phase in {"pending_once", "running"}:
            ready = current.get("canonical_terminal_projection_ready")
            if isinstance(ready, dict) and str(ready.get("summary_id") or "") == summary_id:
                return {"status": str(current.get("status") or status)}
            return {
                "status": str(current.get("status") or status),
                "canonical_terminal_projection_ready": {
                    "summary_id": summary_id,
                    "task_done_ts": str(event.get("ts") or utc_now_iso()),
                    "chat_id": int(event.get("chat_id") or task.get("chat_id") or 0),
                },
            }
        effective = {**result, **current}
        project_id = resolve_project_id({**task, **effective})
        role = str(effective.get("role") or task.get("role") or ("root" if is_root else "child"))
        reason = str(effective.get("reason_code") or event.get("reason_code") or "")
        outcome = completion_status_label(effective, event)
        excerpt = _completion_excerpt(effective)
        details = f'Details: get_task_result(task_id="{tid}")'
        text = (
            f"{outcome}. role={role}; parent={parent_id or 'unknown'}; "
            f"root={root_id}; project={project_id or 'none'}."
        )
        if excerpt:
            text += f" {excerpt}"
        if reason:
            text += f" Reason: {reason}."
        result_ref = {"kind": "task_result", "task_id": tid, "reader": "get_task_result"}
        row = {
            "ts": str(event.get("ts") or effective.get("ts") or utc_now_iso()),
            "direction": "system", "type": "task_summary", "summary_kind": summary_kind,
            "summary_id": summary_id, "task_id": tid,
            "parent_task_id": parent_id, "root_task_id": root_id,
            "project_id": project_id,
            "chat_id": int(event.get("chat_id") or task.get("chat_id") or 0),
            "delegation_role": str(effective.get("delegation_role") or task.get("delegation_role") or ""),
            "role": role, "status": str(effective.get("status") or status),
            "outcome": outcome, "outcome_final": True,
            "outcome_authority": "canonical_task_result_after_finalization",
            "outcome_axes": effective.get("outcome_axes") or event.get("outcome_axes") or {},
            "reason_code": reason, "result_ref": result_ref,
            "text": f"{text} {details}",
        }
        appended = append_canonical_task_summary(drive_root, row)
        if not appended:
            return {"status": str(current.get("status") or status)}
        return {
            "status": str(current.get("status") or status),
            "canonical_terminal_projection": {
                "summary_id": summary_id, "summary_kind": summary_kind,
                "written_at": row["ts"],
            },
            "canonical_terminal_projection_ready": None,
        }

    write_task_result(
        drive_root, tid, status, _field_projector=_append_once,
    )
    return appended


def append_terminal_task_projection(
    drive_root: Any, task_id: str, task: Dict[str, Any], result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> bool:
    """Fail-soft terminal projection; lifecycle cleanup must always continue."""
    try:
        return _append_terminal_task_projection(
            drive_root, task_id, task, result, task_done_event,
        )
    except Exception:
        log.warning("Failed to append canonical terminal projection for %s", task_id, exc_info=True)
        return False


def _completion_excerpt(result: Dict[str, Any]) -> str:
    """One plain-text excerpt for BOTH lifecycle writers (event + task_summary).

    Markdown markers are stripped BEFORE whitespace flattening: the stripper's
    line-anchored heading/list patterns need the original newlines, and a
    flatten-first order would glue a ``##`` mid-line where no pattern (and no
    renderer) can treat it as markup again.
    """
    if str(result.get("terminal_origin") or "") == TERMINAL_ORIGIN_HOST_SALVAGE:
        return ""
    for key in ("summary", "result", "error"):
        text = " ".join(strip_markdown(str(result.get(key) or "")).split())
        if text:
            return text if len(text) <= 240 else text[:239].rstrip() + "…"
    return ""


def _run_lives_in_its_project(
    drive_root: Any, task_id: str, project_id: str, task: Dict[str, Any], result: Dict[str, Any],
) -> bool:
    """Did this run's work actually go into that project's room?

    Two facts answer yes, and only these two. The run was ADDRESSED there —
    admission resolves a registered project's thread, and a scoped run cannot be
    addressed anywhere else. Or the run is BOUND to it, which is how a task that
    started unscoped joins a project mid-flight; a binding re-homes every row it
    already wrote, including the ones written before the project existed.

    Registration alone is not that fact. A run scoped to an id nobody had
    registered yet is admitted to the hidden partition; if that id is registered
    while the run is still going, a room appears its rows never entered, and
    answering "a room exists" is how the reported defect comes back.
    """
    try:
        from ouroboros.projects_registry import get_reserved_project, project_binding_for_task

        chat_id = result.get("chat_id")
        if chat_id is None:
            chat_id = task.get("chat_id")
        project_chat = (get_reserved_project(drive_root, project_id) or {}).get("chat_id")
        if chat_id is not None and project_chat is not None and int(chat_id) == int(project_chat):
            return True
        binding = project_binding_for_task(drive_root, task_id) or {}
        return str(binding.get("project_id") or "") == str(project_id)
    except Exception:
        log.debug("project-room membership check failed for %s", task_id, exc_info=True)
        return False


def enqueue_project_completion_summary(
    drive_root: Any, evt: Dict[str, Any], task_id: str, task: Dict[str, Any],
    result: Dict[str, Any], task_done_event: Dict[str, Any],
) -> bool:
    """Owe Main's one compact row for a terminal non-ephemeral Project root."""
    tid = str(task_id or "").strip()
    task = task if isinstance(task, dict) else {}
    result = result if isinstance(result, dict) else {}
    if not tid or any(
        bool(row.get("_ephemeral") or row.get("ephemeral_decision"))
        for row in (evt, task, result, task_done_event) if isinstance(row, dict)
    ):
        return False
    try:
        from ouroboros.projects_registry import task_presentation_snapshot
        from ouroboros.task_results import resolve_task_lineage
        from ouroboros.task_status import SETTLED_STATUSES
        from supervisor.terminal_delivery import enqueue_terminal_delivery

        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        lineage = resolve_task_lineage(
            tid, metadata=metadata,
            root_task_id=result.get("root_task_id") or task.get("root_task_id"),
            parent_task_id=result.get("parent_task_id") or task.get("parent_task_id"),
            delegation_role=result.get("delegation_role") or task.get("delegation_role"),
            original_task_id=result.get("original_task_id") or task.get("original_task_id"),
            timeout_retry_from=result.get("timeout_retry_from") or task.get("timeout_retry_from"),
        )
        status = str(result.get("status") or task_done_event.get("status") or "").lower()
        if not lineage["is_root_task"] or status not in SETTLED_STATUSES:
            return False
        snapshot = task_presentation_snapshot(
            drive_root, tid, task=task, result=result,
            project_id=str(result.get("project_id") or task.get("project_id") or ""),
        )
        if not snapshot["project_id"] or not snapshot["project_routable"]:
            # Owner decision 3A: a run whose project id was DERIVED from a
            # workspace has no room, so Main stays silent instead of offering an
            # "Open Project" that lands in an empty duplicate of itself. The same
            # holds once a project is deleting or tombstoned.
            return False
        if not _run_lives_in_its_project(drive_root, tid, snapshot["project_id"], task, result):
            # The room exists but holds none of this run's work: its id was only
            # registered AFTER admission, or a mid-flight bind failed fail-soft.
            # Offering "Open the Project" would reproduce the reported defect —
            # a Main row leading into an empty room.
            return False
        excerpt = _completion_excerpt(result)
        event = {
            "type": "send_message", "chat_id": 1, "task_id": tid,
            "text": (f"{snapshot['target_label']} · "
                     f"{completion_status_label(result, task_done_event)}\n"
                     f"{excerpt or 'Open the Project for details.'}"),
            "role": "system", "system_type": "project_completion_summary",
            "delivery_id": f"project-completion:{tid}",
            "progress_meta": {
                "project_id": snapshot["project_id"],
                "project_name": snapshot["project_name"],
                "target_label": snapshot["target_label"], "status": status,
            },
        }
        return bool(enqueue_terminal_delivery(drive_root, event))
    except Exception:
        log.warning("Failed to enqueue Project completion summary for %s", tid, exc_info=True)
        return False


def announce_project_started(
    drive_root: Any, project: Dict[str, Any], task_id: str, *, task: Any = None,
) -> bool:
    """Owe Main's one durable entry row when the AGENT starts a Project.

    Mirrors ``enqueue_project_completion_summary``'s delivery mechanics: the
    same terminal-delivery outbox, with the restart-surviving
    ``delivery_id=project-start:<project_id>`` dedupe as the ONLY dedupe.
    Called exclusively from the agent-initiated creation seams (owner decision
    2=A): the promote_chat_to_task bind and a REAL ensure_project_scope create
    (``created is True`` from ``create_project``). Owner HTTP/API creation and
    manual task-to-project conversion stay silent.
    """
    project = project if isinstance(project, dict) else {}
    pid = str(project.get("id") or "").strip()
    tid = str(task_id or "").strip()
    if not pid:
        return False
    try:
        from ouroboros.projects_registry import task_presentation_snapshot
        from supervisor.terminal_delivery import enqueue_terminal_delivery

        snapshot = task_presentation_snapshot(
            drive_root, tid, task=task if isinstance(task, dict) else None,
            project_id=pid,
        )
        event = {
            "type": "send_message", "chat_id": 1, "task_id": tid,
            "text": (f"{snapshot['target_label']} · Started\n"
                     "Work is running in this Project."),
            "role": "system", "system_type": "project_started",
            "delivery_id": f"project-start:{pid}",
            "progress_meta": {
                "project_id": pid,
                "project_name": snapshot["project_name"],
                "target_label": snapshot["target_label"],
            },
        }
        return bool(enqueue_terminal_delivery(drive_root, event))
    except Exception:
        log.warning("Failed to enqueue Project started row for %s", pid, exc_info=True)
        return False


__all__ = [
    "announce_project_started",
    "append_authored_task_summary",
    "append_chat_annotation",
    "append_canonical_task_summary",
    "append_terminal_task_projection",
    "build_owner_message_ref",
    "chat_annotation_receipt",
    "entry_matches_source_ref",
    "latest_chat_annotations",
    "enqueue_project_completion_summary",
    "completion_status_label",
    "owner_message_ref_is_valid",
    "project_origin_rows",
    "project_recent_dialogue",
    "routing_options_with_labels",
    "routing_target_label",
    "resolve_owner_message_source",
    "source_refs_for_project",
]
