"""Shared Starlette HTTP-API helpers for thin route modules."""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from typing import Any

import anyio
from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.utils import iter_jsonl_objects


_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off"})


from ouroboros.jsonl_tail import (  # noqa: E402
    ARCHIVE_BACKFILL_MAX,
    TAIL_WINDOW_START_BYTES as _TAIL_WINDOW_START_BYTES,  # noqa: F401  (re-exported for gateway/history.py)
)


async def run_sync_to_completion(function, /, *args, **kwargs):
    """Run blocking request work without abandoning its custody on cancellation.

    The caller's cancellation is re-raised only after the one worker settles.
    Admission rollback/publication and materialization therefore retain their
    existing synchronous ownership, even when the HTTP waiter disconnects.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await settle_to_completion(worker)
        raise


async def settle_to_completion(task: asyncio.Task) -> None:
    """Wait until an owned task is done, through the caller's cancellation.

    Its failure is logged, not raised; a cancelled task still raises.
    """
    # Starlette streams use level cancellation; shield that scope while
    # also tolerating repeated raw asyncio Task.cancel() calls.
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        try:
            task.result()
        except Exception:
            logging.getLogger(__name__).debug(
                "Request worker failed while cancellation settled", exc_info=True,
            )


def read_rotated_jsonl_entries(
    live: pathlib.Path,
    archive_dir: pathlib.Path,
    archive_prefix: str,
    want: int,
    counts_toward_quota,
    max_archives: int = ARCHIVE_BACKFILL_MAX,
    *,
    include_gaps: bool = False,
) -> list | tuple[list, set[str]]:
    """Bounded, rotation-aware read of one JSONL log (v6.90.x P2).

    The reader itself lives in ``ouroboros/jsonl_tail.py`` (one bounded
    filtered tail for the endpoints AND context assembly); this wrapper keeps
    the gateway call shape and its parser seam (see above).
    """
    from ouroboros.jsonl_tail import read_rotated_jsonl_entries as _read

    return _read(
        live, archive_dir, archive_prefix, want, counts_toward_quota, max_archives,
        include_gaps=include_gaps, iter_objects=iter_jsonl_objects,
    )


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
    "stage_initial_task_attachments", "run_sync_to_completion", "settle_to_completion",
)
