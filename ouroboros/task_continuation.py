"""Typed durable review-continuation payloads for blocked/interrupted tasks."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import atomic_write_json, replace_atomic, utc_now_iso

_CONTINUATION_DIR_RELPATH = "state/review_continuations"
_CORRUPT_DIR_NAME = "corrupt"


class ContinuationCorruptError(ValueError):
    """Raised when a continuation file exists but is malformed/corrupt."""


@dataclass
class ReviewContinuation:
    task_id: str
    source: str
    stage: str
    repo_key: str = ""
    tool_name: str = ""
    attempt: int = 0
    commit_message: str = ""
    block_reason: str = ""
    block_details: str = ""
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    advisory_findings: List[Dict[str, Any]] = field(default_factory=list)
    # Written by bodies before the obligation layer retired; read only as recorded open work (``open_work_matcher``).
    obligation_ids: List[str] = field(default_factory=list)
    open_obligations: List[Dict[str, Any]] = field(default_factory=list)
    readiness_warnings: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    task_type: str = ""
    parent_task_id: str = ""
    created_ts: str = ""
    updated_ts: str = ""
    # The attempt's review ledger record, where every seat's answer stays as written; "" = unknown (an older file).
    review_record_id: str = ""


def continuation_dir(drive_root: Any) -> pathlib.Path:
    path = pathlib.Path(drive_root) / _CONTINUATION_DIR_RELPATH
    path.mkdir(parents=True, exist_ok=True)
    return path


def corrupt_continuation_dir(drive_root: Any) -> pathlib.Path:
    path = continuation_dir(drive_root) / _CORRUPT_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def continuation_path(drive_root: Any, task_id: str) -> pathlib.Path:
    return continuation_dir(drive_root) / f"{task_id}.json"


def load_review_continuation(drive_root: Any, task_id: str) -> Optional[ReviewContinuation]:
    path = continuation_path(drive_root, task_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContinuationCorruptError(f"{path.name}: invalid JSON: {exc}") from exc
    return _continuation_from_dict(data, expected_task_id=task_id)


def list_review_continuations(drive_root: Any) -> Tuple[List[ReviewContinuation], List[str]]:
    continuations: List[ReviewContinuation] = []
    corrupt: List[str] = []
    for path in sorted(continuation_dir(drive_root).glob("*.json")):
        task_id = path.stem
        try:
            item = load_review_continuation(drive_root, task_id)
        except ContinuationCorruptError as exc:
            corrupt.append(str(exc))
            continue
        if item is not None:
            continuations.append(item)
    corrupt.extend(_list_quarantined_corrupt_messages(drive_root))
    return continuations, corrupt


def save_review_continuation(
    drive_root: Any,
    continuation: ReviewContinuation,
    *,
    expect_task_id: str = "",
) -> ReviewContinuation:
    task_id = str(continuation.task_id or "").strip()
    if not task_id:
        raise ValueError("ReviewContinuation.task_id is required")
    if expect_task_id and task_id != str(expect_task_id).strip():
        raise ValueError(
            f"Continuation ownership mismatch: payload task_id={task_id!r}, expected={expect_task_id!r}"
        )

    path = continuation_path(drive_root, task_id)
    existing = None
    if path.exists():
        try:
            existing = load_review_continuation(drive_root, task_id)
        except ContinuationCorruptError as exc:
            _quarantine_corrupt_continuation(drive_root, task_id, reason=str(exc))
            existing = None
    now_ts = utc_now_iso()
    if not continuation.created_ts:
        continuation.created_ts = existing.created_ts if existing else now_ts
    continuation.updated_ts = now_ts

    payload = asdict(continuation)
    atomic_write_json(path, payload)
    return continuation


def clear_review_continuation(drive_root: Any, task_id: str) -> bool:
    path = continuation_path(drive_root, task_id)
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


_ARCHIVED_DIR_NAME = "archived"

# A continuation of a SETTLED task younger than this stays in context: it is the
# designed cross-task resume pointer (the owning task died with the commit still
# blocked; a follow-up picks it up from the Review Continuity section). Older
# than this, un-resumed, it is history riding into every new task's prompt.
RETIRE_SETTLED_CONTINUATION_AFTER_DAYS = 7


def archived_continuation_dir(drive_root: Any) -> pathlib.Path:
    path = continuation_dir(drive_root) / _ARCHIVED_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _continuation_age_days(item: ReviewContinuation) -> float:
    import datetime as _dt

    ts = str(item.updated_ts or item.created_ts or "").strip()
    if not ts:
        return float("inf")  # undatable record: old by construction
    try:
        then = _dt.datetime.fromisoformat(ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(tz=_dt.timezone.utc) - then).total_seconds() / 86400.0
    except ValueError:
        return float("inf")


def retire_settled_continuations(
    drive_root: Any,
    *,
    is_settled: Any,
    has_open_work: Any = None,
    min_age_days: float = RETIRE_SETTLED_CONTINUATION_AFTER_DAYS,
) -> List[str]:
    """Archive continuations whose owning task SETTLED and that sat un-resumed.

    A continuation exists so a follow-up can resume a blocked/interrupted review
    — so a FRESH record of a settled task stays. But nothing clears the file
    when a different task later lands the same work, so stale records rode into
    every new task's Review Continuity section indefinitely (live: Aug-5 records
    injected for days). Retirement MOVES a settled-and-older-than-``min_age_days``
    file to ``state/review_continuations/archived/`` — durable, never deleted
    (P1) — and returns the retired task_ids. ``is_settled(task_id) -> bool`` is
    supplied by the caller so this module stays free of task-result imports; any
    error leaves the continuation in place (fail-open: context noise over lost
    review state).

    Age alone never proves the blocked review work landed. When the caller can
    check, it supplies ``has_open_work(continuation) -> bool``: a settled-and-old
    continuation that still records open work is kept in place — it remains the
    designed resume pointer for genuinely unresolved work (P1/P3).
    """
    retired: List[str] = []
    for path in sorted(continuation_dir(drive_root).glob("*.json")):
        task_id = path.stem
        try:
            item = load_review_continuation(drive_root, task_id)
            if item is None or _continuation_age_days(item) < min_age_days:
                continue
            if not is_settled(task_id):
                continue
            if has_open_work is not None and has_open_work(item):
                continue  # recorded review work still open: keep the resume pointer
            target = archived_continuation_dir(drive_root) / path.name
            if target.exists():
                target = target.with_name(f"{path.stem}.{_safe_ts_token(utc_now_iso())}.json")
            path.rename(target)
            retired.append(task_id)
        except ContinuationCorruptError:
            continue  # the corrupt-quarantine path owns malformed files
        except Exception:
            continue
    return retired


def build_review_continuation(
    task: Dict[str, Any],
    attempt: Any,
    *,
    source: str,
    warning: str = "",
) -> Optional[ReviewContinuation]:
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return None

    warnings = [str(item) for item in (getattr(attempt, "readiness_warnings", []) or []) if str(item).strip()]
    if warning:
        warnings.append(str(warning))
    if attempt is None:
        if source != "task_exception" or not warnings:
            return None
        return ReviewContinuation(
            task_id=task_id,
            source=source,
            stage=source,
            readiness_warnings=warnings,
            warnings=warnings,
            task_type=str(task.get("type") or ""),
            parent_task_id=str(task.get("parent_task_id") or ""),
        )

    return ReviewContinuation(
        task_id=task_id,
        source=source,
        stage=str(getattr(attempt, "phase", "") or source),
        repo_key=str(getattr(attempt, "repo_key", "") or ""),
        tool_name=str(getattr(attempt, "tool_name", "") or ""),
        attempt=int(getattr(attempt, "attempt", 0) or 0),
        commit_message=str(getattr(attempt, "commit_message", "") or ""),
        block_reason=str(getattr(attempt, "block_reason", "") or ""),
        block_details=str(getattr(attempt, "block_details", "") or ""),
        critical_findings=list(getattr(attempt, "critical_findings", []) or []),
        advisory_findings=list(getattr(attempt, "advisory_findings", []) or []),
        readiness_warnings=warnings,
        degraded_reasons=list(getattr(attempt, "degraded_reasons", []) or []),
        warnings=[str(item) for item in warnings if str(item).strip()],
        task_type=str(task.get("type") or ""),
        parent_task_id=str(task.get("parent_task_id") or ""),
        review_record_id=str(getattr(attempt, "review_record_id", "") or ""),
    )


def capture_review_continuation_from_state(
    drive_root: Any,
    task: Dict[str, Any],
    *,
    source: str,
    warning: str = "",
    repo_dir: Any = None,
) -> Optional[ReviewContinuation]:
    from ouroboros.review_state import load_state, make_repo_key

    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return None

    state = load_state(pathlib.Path(drive_root))
    repo_key = make_repo_key(pathlib.Path(repo_dir)) if repo_dir else ""
    attempt = (
        state.latest_attempt_for(task_id=task_id, repo_key=repo_key or None)
        if repo_key
        else state.latest_attempt_for(task_id=task_id)
    )
    continuation = build_review_continuation(task, attempt, source=source, warning=warning)
    if continuation is None:
        return None
    return save_review_continuation(drive_root, continuation, expect_task_id=task_id)


def _continuation_from_dict(data: Dict[str, Any], *, expected_task_id: str = "") -> ReviewContinuation:
    if not isinstance(data, dict):
        raise ContinuationCorruptError("Continuation payload must be a JSON object")
    task_id = str(data.get("task_id", "") or "").strip()
    if not task_id:
        raise ContinuationCorruptError("Continuation missing task_id")
    if expected_task_id and task_id != expected_task_id:
        raise ContinuationCorruptError(
            f"Continuation task_id mismatch: {task_id!r} != {expected_task_id!r}"
        )
    source = str(data.get("source", "") or "").strip()
    stage = str(data.get("stage", "") or "").strip()
    if not source or not stage:
        raise ContinuationCorruptError("Continuation missing source/stage")

    return ReviewContinuation(
        task_id=task_id,
        source=source,
        stage=stage,
        repo_key=str(data.get("repo_key", "") or ""),
        tool_name=str(data.get("tool_name", "") or ""),
        attempt=int(data.get("attempt", 0) or 0),
        commit_message=str(data.get("commit_message", "") or ""),
        block_reason=str(data.get("block_reason", "") or ""),
        block_details=str(data.get("block_details", "") or ""),
        critical_findings=list(data.get("critical_findings") or []),
        advisory_findings=list(data.get("advisory_findings") or []),
        obligation_ids=[str(x) for x in (data.get("obligation_ids") or [])],
        open_obligations=[
            item if isinstance(item, dict) else {"value": str(item)}
            for item in (data.get("open_obligations") or [])
        ],
        readiness_warnings=[str(x) for x in (data.get("readiness_warnings") or [])],
        degraded_reasons=[str(x) for x in (data.get("degraded_reasons") or [])],
        warnings=[str(x) for x in (data.get("warnings") or [])],
        task_type=str(data.get("task_type", "") or ""),
        parent_task_id=str(data.get("parent_task_id", "") or ""),
        created_ts=str(data.get("created_ts", "") or ""),
        updated_ts=str(data.get("updated_ts", "") or ""),
        review_record_id=str(data.get("review_record_id", "") or ""),
    )


def _quarantine_corrupt_continuation(drive_root: Any, task_id: str, *, reason: str) -> None:
    src = continuation_path(drive_root, task_id)
    if not src.exists():
        return
    stamp = _safe_ts_token(utc_now_iso())
    corrupt_dir = corrupt_continuation_dir(drive_root)
    archived = corrupt_dir / f"{task_id}.{stamp}.json"
    note = archived.with_suffix(".txt")
    replace_atomic(src, archived)
    note.write_text(reason.strip() or "corrupt continuation quarantined", encoding="utf-8")


def _list_quarantined_corrupt_messages(drive_root: Any) -> List[str]:
    corrupt_dir = continuation_dir(drive_root) / _CORRUPT_DIR_NAME
    if not corrupt_dir.exists():
        return []
    messages: List[str] = []
    for path in sorted(corrupt_dir.glob("*.json")):
        note = path.with_suffix(".txt")
        reason = ""
        if note.exists():
            try:
                reason = note.read_text(encoding="utf-8").strip()
            except Exception:
                reason = ""
        msg = f"{path.name}: quarantined corrupt continuation"
        if reason:
            msg = f"{msg}: {reason}"
        messages.append(msg)
    return messages


def _safe_ts_token(ts: str) -> str:
    token = str(ts or "").strip()
    if not token:
        return "unknown"
    return token.replace(":", "").replace("-", "").replace("+", "_")


def open_work_matcher(state: Any) -> Any:
    """Build ``has_open_work(continuation)``: a continuation carrying an obligation mark written before the
    obligation layer retired (``obligation_ids`` or its ``open_obligations`` snapshot) records open work until a
    commit lands in its repository after it was last written, which is when the layer cleared those obligations.
    A continuation that holds only findings follows the age rule. ``state`` is duck-typed (``landed_after``) so
    this module keeps zero review-state imports."""

    def _has_open_work(item: Any) -> bool:
        if not (item.obligation_ids or item.open_obligations):
            return False
        return not state.landed_after(item.repo_key, item.updated_ts or item.created_ts)

    return _has_open_work


def retire_settled_continuations_for_context(
    drive_root: Any, state: Any, load_result: Any
) -> List[str]:
    """Retire aged settled continuations, keeping unresolved review work live.

    The Review Continuity preamble of ``build_review_context``: ``state``
    supplies the review ledger and ``load_result(task_id)`` the task
    result dict — both injected so this module stays free of review-state and
    task-result imports. A settled continuation that still records open work
    (``open_work_matcher``) is not archived by age (P1/P3).
    """
    from ouroboros.task_status import SETTLED_STATUSES

    def _is_settled(tid: str) -> bool:
        status = str((load_result(tid) or {}).get("status") or "")
        return status.strip().lower() in SETTLED_STATUSES

    return retire_settled_continuations(
        drive_root, is_settled=_is_settled, has_open_work=open_work_matcher(state))
