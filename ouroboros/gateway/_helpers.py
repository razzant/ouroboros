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


def read_rotated_jsonl_entries(
    live: pathlib.Path,
    archive_dir: pathlib.Path,
    archive_prefix: str,
    want: int,
    counts_toward_quota,
    max_archives: int = 3,
) -> list:
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
    except OSError:
        size = 0
    window = _TAIL_WINDOW_START_BYTES
    while True:
        if window >= size:
            live_entries = list(iter_jsonl_objects(live))
            break
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
            archive_entries = list(iter_jsonl_objects(archive_path))
        except Exception:
            continue
        chosen.append(archive_entries)
        collected += sum(1 for entry in archive_entries if counts_toward_quota(entry))
    ordered: list = []
    for archive_entries in reversed(chosen):  # oldest chosen archive first
        ordered.extend(archive_entries)
    ordered.extend(live_entries)
    return ordered


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


def wire_error_code(value: Any) -> str:
    """THE wire spelling of a typed refusal code: lower-case, bounded, stripped.

    The authorities disagree on case by construction — a code that is a Python
    CONSTANT reads naturally as ``REMOTE_TRANSPORT_UNAVAILABLE``
    (``workspace_admission``) while a code the transport raises is already
    ``remote_transport_unavailable``. Consumers compare it case-SENSITIVELY (the
    browser's ``remote_task_state.js``, the CLI's exit-code sets), so the
    normalization has to happen ONCE, at the boundary where the code becomes a
    wire field. Every gateway module that writes ``error_code`` goes through here
    rather than sprinkling ``.lower()`` at the call sites, which is how one of the
    two spellings escaped to the browser in the first place.
    """
    return str(value or "").strip().lower()[:128]


__all__ = (
    "coerce_bool", "coerce_int", "iter_jsonl_objects", "json_error", "json_exception",
    "read_rotated_jsonl_entries", "request_json_or", "request_drive_root", "request_repo_dir",
    "wire_error_code",
)
