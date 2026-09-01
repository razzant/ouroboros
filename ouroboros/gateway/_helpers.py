"""Shared Starlette HTTP-API helpers for thin route modules."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.utils import iter_jsonl_objects


_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off"})


_TAIL_WINDOW_START_BYTES = 512 * 1024


def _read_jsonl_segment_with_gaps(
    path: pathlib.Path,
    *,
    tail_bytes: int | None = None,
) -> tuple[list, set[str]]:
    """Read one JSONL segment while retaining truthful parse/read-gap facts.

    ``iter_jsonl_objects`` intentionally skips malformed rows because most
    callers are best-effort telemetry readers.  Gateway history is different:
    it publishes a completeness claim, so a skipped row must remain visible as
    a bounded read-gap fact even when the valid rows can still be rendered.
    """
    path = pathlib.Path(path)
    try:
        path.stat()
    except FileNotFoundError:
        return [], set()
    except OSError:
        return [], {"unreadable_source"}

    gaps: set[str] = set()
    try:
        entries = list(
            iter_jsonl_objects(path, tail_bytes=tail_bytes, gap_reasons=gaps)
        )
    except OSError:
        gaps.add("unreadable_source")
    return entries, gaps


def read_rotated_jsonl_entries(
    live: pathlib.Path,
    archive_dir: pathlib.Path,
    archive_prefix: str,
    want: int,
    counts_toward_quota,
    max_archives: int = 3,
    *,
    include_gaps: bool = False,
) -> list | tuple[list, set[str]]:
    """Bounded, rotation-aware read of one JSONL log (v6.90.x P2, built on the
    ``iter_jsonl_objects(tail_bytes=...)`` bounded-read SSOT).

    The live file is read from a byte tail that starts at 512KB and DOUBLES until
    the FILTERED quota is satisfied (rows for which ``counts_toward_quota`` is
    true reach ``want``) or the window covers the whole file — the degenerate
    case equals today's full read, so a quota the file cannot satisfy costs one
    full pass, never an infinite loop. Rotated ``archive/<prefix>_*.jsonl``
    segments are then backfilled newest-first until the quota is met, bounded to
    ``max_archives`` files, and everything is reassembled chronologically
    (oldest chosen archive -> live window). The backfill is bounded to the
    ``max_archives`` newest archives: older segments are NOT consulted by this
    reader (they stay durable on disk for full-history consumers)."""
    live = pathlib.Path(live)
    try:
        size = live.stat().st_size
    except FileNotFoundError:
        size = 0
    except OSError:
        size = 0
    window = _TAIL_WINDOW_START_BYTES
    gaps: set[str] = set()
    while True:
        if window >= size:
            if include_gaps:
                live_entries, live_gaps = _read_jsonl_segment_with_gaps(live)
                gaps.update(live_gaps)
            else:
                live_entries = list(iter_jsonl_objects(live))
            break
        if include_gaps:
            live_entries, live_gaps = _read_jsonl_segment_with_gaps(live, tail_bytes=window)
            gaps.update(live_gaps)
        else:
            live_entries = list(iter_jsonl_objects(live, tail_bytes=window))
        if sum(1 for entry in live_entries if counts_toward_quota(entry)) >= want:
            break
        window *= 2
    collected = sum(1 for entry in live_entries if counts_toward_quota(entry))
    try:
        archives = sorted(
            archive_dir.glob(f"{archive_prefix}_*.jsonl"), key=lambda p: p.name, reverse=True
        )
    except Exception:
        archives = []
    chosen: list = []
    for archive_path in archives:
        if collected >= want or len(chosen) >= max_archives:
            break
        try:
            if include_gaps:
                archive_entries, archive_gaps = _read_jsonl_segment_with_gaps(archive_path)
                gaps.update(archive_gaps)
            else:
                archive_entries = list(iter_jsonl_objects(archive_path))
        except Exception:
            gaps.add("unreadable_source")
            continue
        chosen.append(archive_entries)
        collected += sum(1 for entry in archive_entries if counts_toward_quota(entry))
    ordered: list = []
    for archive_entries in reversed(chosen):  # oldest chosen archive first
        ordered.extend(archive_entries)
    ordered.extend(live_entries)
    return (ordered, gaps) if include_gaps else ordered


def request_drive_root(request: Request) -> pathlib.Path:
    """Drive root pinned on ``request.app.state`` or the configured default."""
    from ouroboros.config import DATA_DIR
    state = getattr(request.app, "state", None)
    drive_root = getattr(state, "drive_root", None) if state is not None else None
    return pathlib.Path(drive_root) if drive_root is not None else pathlib.Path(DATA_DIR)


def request_repo_dir(request: Request) -> pathlib.Path:
    """Repo dir pinned on ``request.app.state`` or the configured default."""
    from ouroboros.config import REPO_DIR
    state = getattr(request.app, "state", None)
    repo_dir = getattr(state, "repo_dir", None) if state is not None else None
    return pathlib.Path(repo_dir) if repo_dir is not None else pathlib.Path(REPO_DIR)


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Best-effort bool coercion accepting common HTTP truthy/falsy literals."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False
    return default


def coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion. Returns ``default`` on parse failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def request_json_or(
    request: Request,
    default: Any,
    *,
    exceptions: tuple[type[BaseException], ...] = (json.JSONDecodeError, ValueError),
) -> Any:
    try:
        return await request.json()
    except exceptions:
        return default


def json_error(message: str, status: int = 500, **extra: Any) -> JSONResponse:
    """``JSONResponse({"error": message, **extra}, status_code=status)``."""
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return JSONResponse(payload, status_code=status)


def json_exception(exc: BaseException, status: int = 500) -> JSONResponse:
    return json_error(str(exc), status)


def stage_initial_task_attachments(
    drive_root: pathlib.Path,
    task_id: str,
    attachments: list,
    *,
    allow_partial: bool,
) -> tuple[list, JSONResponse | None]:
    """Stage one API task's complete manifest or return its atomic refusal."""

    from ouroboros.artifacts import (
        attachment_manifest_has_rejections,
        remove_staged_attachments,
        stage_task_attachments,
    )

    try:
        manifest = stage_task_attachments(drive_root, task_id, attachments)
    except Exception as exc:
        return [], json_exception(exc, 503)
    if not attachment_manifest_has_rejections(manifest) or allow_partial:
        return manifest, None
    remove_staged_attachments(manifest)
    return manifest, JSONResponse({
        "ok": False,
        "task_id": task_id,
        "status": "rejected",
        "reason_code": "attachment_admission_rejected",
        "error": (
            "Task was not scheduled because one or more declared attachments "
            "could not be staged and this request asked for atomic admission "
            "(allow_partial_attachments=false). Retry with corrected "
            "attachments, or omit the flag to stage the good ones partially."
        ),
        "attachment_manifest": manifest,
    }, status_code=422)


__all__ = (
    "coerce_bool", "coerce_int", "iter_jsonl_objects", "json_error", "json_exception",
    "read_rotated_jsonl_entries", "request_json_or", "request_drive_root", "request_repo_dir",
    "stage_initial_task_attachments",
)
