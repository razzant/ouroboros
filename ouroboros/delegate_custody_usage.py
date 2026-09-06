"""Pure usage and terminal-state projections for delegated-run custody."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)


@contextmanager
def project_retirement_lock(drive_root: Any, project_id: str):
    """Cross-process lock for one project's settlement/retirement decision."""
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock, release_exclusive_file_lock,
    )

    digest = hashlib.sha256(str(project_id or "").encode("utf-8")).hexdigest()[:24]
    lock_path = pathlib.Path(drive_root) / "state" / "delegate_project_retirements" / f"{digest}.lock"
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=20.0, stale_sec=120.0, owner_aware_stale=True,
    )
    if fd is None:
        raise TimeoutError("project retirement lock is unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def summary_of(detail: Dict[str, Any]) -> Dict[str, Any]:
    return detail.get("summary") if isinstance(detail.get("summary"), dict) else {}


def disclosed_spend(summary: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """The cash the harness reported AND whether it is settled — never one without both.

    ``spendUsd`` is only half the disclosure: the engine populates the sibling
    ``spendEstimated``. Reading the amount alone makes an estimate
    indistinguishable from settled cash, so callers receive the pair atomically.
    """
    raw = summary.get("spendUsd")
    if raw is None:
        return None, False
    try:
        return float(raw), summary.get("spendEstimated") is True
    except (TypeError, ValueError):
        return None, False


def disclosed_tokens(raw: Any) -> Optional[int]:
    """A reported token count, or ``None`` when the harness reported nothing.

    The control schema keeps token counts null until a harness reports them;
    converting absence to zero would erase that distinction.
    """
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def is_terminal(detail: Dict[str, Any], terminal_states: frozenset[str]) -> bool:
    return str(summary_of(detail).get("state") or "") in terminal_states


def complete_custody_rows(path, marker: str, *, started_type: str = ""):
    """Every custody row, or ``None`` when the log's view is INCOMPLETE.

    The lenient reader skips unreadable lines to keep liveness surfaces
    working; an authority decision (removing a shared project) must instead
    fail closed: a marker-bearing line that cannot decode as strict UTF-8 or
    parse as a row, or a STARTED row missing its run identity, means a
    sibling's state may be invisible - no complete view exists. Streamed
    line-by-line (event logs grow to hundreds of MB). Chain-aware (CPL4-C1):
    reads the rotated ``archive/events_*.jsonl`` segments before the live
    file — live-first open + inode dedup keeps a racing rotation from hiding
    rows — and anything in the chain that exists but cannot be READ (segment,
    live file, or the archive directory itself) is an incomplete view, not an
    empty one: the STRICT chain reader turns each of those into a typed
    ``JsonlChainUnreadable`` this authority answers with ``None``."""
    import json

    from ouroboros.utils import JsonlChainUnreadable, jsonl_chain_handles

    try:
        with jsonl_chain_handles(path, strict=True) as handles:
            rows = []
            for _, handle in handles:
                for raw in handle:
                    if marker.encode("ascii") not in raw:
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8", errors="strict"))
                    except (ValueError, UnicodeDecodeError):
                        return None
                    if not isinstance(row, dict):
                        return None
                    if not str(row.get("type") or "").startswith(marker):
                        continue  # a valid row of another event family
                    if started_type and row.get("type") == started_type and not str(row.get("run_id") or ""):
                        return None
                    rows.append(row)
            return rows
    except (JsonlChainUnreadable, OSError):
        return None


# ---- reviewer usage observation (one llm_usage row per physical send) ----

_EMITTED_SESSION_USAGE: Dict[str, float] = {}  # delegated run id -> first llm_usage emission


def session_usage_once(run_id: str) -> bool:
    """One llm_usage row per delegated run, across executors (process-local)."""
    key = str(run_id or "")
    if key in _EMITTED_SESSION_USAGE:
        return False
    if key:
        _EMITTED_SESSION_USAGE.clear() if len(_EMITTED_SESSION_USAGE) >= 256 else None
        _EMITTED_SESSION_USAGE[key] = time.monotonic()
    return True


def observe_review_usage(observer: Any, usage: Optional[Dict[str, Any]]) -> None:
    """Hand one row per ledger attempt to the reviewer usage observer."""
    if observer is None:
        return
    row = dict(usage or {})
    attempt_ids = [str(value) for value in (row.get("ledger_attempt_ids") or []) if value]
    for attempt_id in attempt_ids[:-1]:
        observer({
            "resolved_model": row.get("resolved_model"), "provider": row.get("provider"),
            "ledger_attempt_ids": [attempt_id],
        })
    if attempt_ids:
        row["ledger_attempt_ids"] = attempt_ids[-1:]
    observer(row)


def observe_failed_review_send(observer: Any, exc: BaseException) -> None:
    """Rows for every physically dispatched attempt behind a failed reviewer send."""
    from ouroboros.usage_accounting import (
        POSITIVE_PHYSICAL_ATTEMPT_STATES, _drive_root, _final_rows, _locked,
        _read_records_locked_cached, current_usage_scope,
    )

    capture = getattr(exc, "physical_attempt_capture", None)
    attempt_ids = [str(value) for value in (getattr(exc, "ledger_attempt_ids", None) or []) if value]
    capture_id = str(getattr(capture, "attempt_id", "") or "")
    if capture_id and capture_id not in attempt_ids:
        attempt_ids.append(capture_id)
    rows: Dict[str, Dict[str, Any]] = {}
    try:
        scope = current_usage_scope()
        root = _drive_root(getattr(scope, "drive_root", None))
        with _locked(root):
            finals = _final_rows(_read_records_locked_cached(root))
        rows = {attempt_id: finals[attempt_id] for attempt_id in attempt_ids if attempt_id in finals}
    except Exception:
        log.debug("failed to resolve review attempt states", exc_info=True)
    capture_state = str(getattr(capture, "state", "") or "")
    for attempt_id in attempt_ids:
        row = rows.get(attempt_id, {})
        state = str(row.get("state") or (capture_state if attempt_id == capture_id else ""))
        if state not in POSITIVE_PHYSICAL_ATTEMPT_STATES and not (
            not rows and attempt_id != capture_id
        ):
            continue
        observe_review_usage(observer, {
            "resolved_model": str(row.get("model") or getattr(capture, "model", "") or ""),
            "provider": str(row.get("provider") or getattr(capture, "provider", "") or ""),
            "ledger_attempt_ids": [attempt_id],
        })
