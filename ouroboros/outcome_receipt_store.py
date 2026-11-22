"""Durable verification-receipt storage and split-root replica reads."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros import _outcome_receipts
from ouroboros.task_results import validate_task_id
from ouroboros.platform_layer import (
    acquire_exclusive_file_lock,
    release_exclusive_file_lock,
)
from ouroboros.utils import (
    iter_jsonl_objects,
    json_dumps_utf8,
    jsonl_append_lock_path,
    write_text_atomic,
)

log = logging.getLogger(__name__)

# The one canonical zero-run decision vocabulary (P7 SSOT), split by contract:
# WRITE — what an actor may still declare: it did not finish (incomplete) or
# cannot know (unknown). A zero-run "complete" is unverifiable self-report and
# stopped being writable under the charter (owner 2026-08-28).
# READ — hydration additionally keeps historical "complete" receipts valid:
# they still fence a second physical start on the same actor; the terminal
# projection degrades them to unknown-with-disclosure instead of clean.
ZERO_RUN_WRITE_DECISIONS = ("incomplete", "unknown")
ZERO_RUN_READ_DECISIONS = frozenset({"complete", *ZERO_RUN_WRITE_DECISIONS})


def terminal_zero_run_receipt(
    receipt: Any, *, gap_reasons: Optional[set[str]] = None,
) -> bool:
    """Validate the exact terminal no-physical-run receipt written by the host."""
    if not isinstance(receipt, dict) or str(
        receipt.get("contract_kind") or ""
    ) != "delegation_zero_run":
        return False
    valid = (
        receipt.get("zero_run") is True
        and str(receipt.get("status") or "") == "declared"
        and str(receipt.get("zero_run_decision") or "") in ZERO_RUN_READ_DECISIONS
        and bool(str(receipt.get("zero_run_basis") or "").strip())
        and receipt.get("physical_run_started") is False
    )
    if not valid and gap_reasons is not None:
        gap_reasons.add("delegation_zero_run_receipt_invalid")
    return valid


def verification_receipts_path(
    drive_root: Any, task_id: str, *, create: bool = False,
) -> pathlib.Path:
    """Return the task's durable verification-receipt JSONL path."""

    safe = validate_task_id(task_id)
    artifact_dir = pathlib.Path(drive_root) / "task_results" / "artifacts" / safe
    if create:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / "verification_receipts.jsonl"


def is_verification_receipts_path(
    drive_root: Any, task_id: str, candidate: Any,
) -> bool:
    """Whether ``candidate`` is this task's receipt authority file.

    Receipt replicas share the task-artifact directory for custody, but they are
    reconciled by :func:`publish_verification_receipt_union`, never by generic
    artifact copying.  Keep the identity test beside the path SSOT so live reads,
    final copy-back and artifact collection cannot derive competing spellings.
    """

    try:
        expected = verification_receipts_path(
            drive_root, task_id, create=False,
        ).resolve(strict=False)
        return pathlib.Path(candidate).resolve(strict=False) == expected
    except (OSError, TypeError, ValueError):
        return False


def append_verification_receipt(
    drive_root: Any, task_id: str, receipt: Dict[str, Any],
) -> bool:
    """Append a receipt and report whether its durable custody succeeded."""

    try:
        from ouroboros.utils import append_jsonl

        return bool(
            append_jsonl(
                verification_receipts_path(drive_root, task_id, create=True),
                receipt,
                ensure_record_boundary=True,
                require_lock=True,
            )
        )
    except Exception:
        log.warning(
            "Failed to append verification receipt for task %s", task_id,
            exc_info=True,
        )
        return False


def read_verification_receipts(
    drive_root: Any, task_id: str, *, gap_reasons: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        path = verification_receipts_path(drive_root, task_id, create=False)
        if gap_reasons is not None:
            try:
                path.stat()
            except FileNotFoundError:
                return []
        elif not path.exists():
            return []
        if gap_reasons is None:
            # Preserve the historical all-or-nothing read used by observational
            # acceptance/ledger callers. Gap-aware authority reads opt into the
            # tolerant iterator below so partial green rows never become truth.
            return _outcome_receipts.read_receipts(path)
        return list(iter_jsonl_objects(path, gap_reasons=gap_reasons))
    except Exception:
        if gap_reasons is not None:
            gap_reasons.add("verification_receipts_unreadable")
        log.warning(
            "Failed to read verification receipts for task %s", task_id,
            exc_info=True,
        )
        return []


def merge_verification_receipts(
    *groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union replicas chronologically with exact-row de-duplication.

    Receipt reconciliation is order-sensitive: a later pass may clear an older
    failure, while a later failure must stay red.  Split-root replicas therefore
    cannot use caller/source order as chronology.  Host receipts carry ISO UTC
    ``ts`` values; undated legacy rows stay stably before dated rows.
    """

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for receipt in group if isinstance(group, list) else []:
            if not isinstance(receipt, dict):
                continue
            marker = json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(dict(receipt))
    indexed = list(enumerate(merged))
    indexed.sort(key=lambda item: (
        1 if str(item[1].get("ts") or "").strip() else 0,
        str(item[1].get("ts") or "").strip(),
        item[0],
    ))
    return [receipt for _index, receipt in indexed]


def publish_verification_receipt_union(
    canonical_root: Any, task_id: str, replica_root: Any,
) -> bool:
    """Publish one receipt replica without racing canonical appenders.

    ``append_jsonl`` serializes receipt appends through a path-derived sidecar.
    Whole-file reconciliation must use that exact lock too: otherwise an append
    that lands after the canonical read but before the atomic replace is erased.
    The replica may continue appending independently; a later publication will
    pick up those rows, while the canonical store is re-read under its lock.
    """

    src = verification_receipts_path(replica_root, task_id, create=False)
    if not src.is_file():
        return False
    dest = verification_receipts_path(canonical_root, task_id, create=True)
    if dest.exists() and src.samefile(dest):
        return True

    replica_gaps: set[str] = set()
    replica_receipts = read_verification_receipts(
        replica_root, task_id, gap_reasons=replica_gaps,
    )
    if replica_gaps:
        log.warning(
            "Skipped receipt publication for task %s due to replica read gaps: %s",
            task_id, ",".join(sorted(replica_gaps)),
        )
        return False

    lock_path = jsonl_append_lock_path(dest)
    lock_fd = acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=2.0,
        stale_sec=10.0,
        poll_sec=0.01,
        owner_aware_stale=True,
    )
    if lock_fd is None:
        log.warning(
            "Skipped receipt publication for task %s: canonical append lock timed out",
            task_id,
        )
        return False
    try:
        canonical_gaps: set[str] = set()
        canonical_receipts = read_verification_receipts(
            canonical_root, task_id, gap_reasons=canonical_gaps,
        )
        if canonical_gaps:
            log.warning(
                "Skipped receipt publication for task %s due to canonical read gaps: %s",
                task_id, ",".join(sorted(canonical_gaps)),
            )
            return False
        merged = merge_verification_receipts(
            replica_receipts,
            canonical_receipts,
        )
        content = "".join(
            json_dumps_utf8(row, ensure_ascii=False) + "\n" for row in merged
        )
        from ouroboros.task_custody import fence_publication

        fence_publication()  # after the lock wait: a closed publication generation replaces nothing
        write_text_atomic(dest, content)
        return True
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def read_verification_receipts_from_roots(
    roots: List[Any], task_id: str, *, gap_reasons: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Read every distinct receipt replica, preserving supplied root order."""

    groups: List[List[Dict[str, Any]]] = []
    seen_roots: set[str] = set()
    for root in roots if isinstance(roots, list) else []:
        if root in (None, ""):
            continue
        try:
            path = pathlib.Path(root).resolve(strict=False)
        except Exception:
            if gap_reasons is not None:
                gap_reasons.add("verification_receipts_root_unavailable")
            continue
        marker = str(path)
        if marker in seen_roots:
            continue
        seen_roots.add(marker)
        groups.append(
            read_verification_receipts(
                path, task_id, gap_reasons=gap_reasons,
            )
        )
    return merge_verification_receipts(*groups)


def read_context_verification_receipts(
    ctx: Any, task_id: str, *, fallback_root: Any = None, gap_reasons: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Read an active actor's local and canonical receipt replicas."""

    roots: List[Any] = [getattr(ctx, "drive_root", None)]
    try:
        from ouroboros.tool_access import canonical_data_root

        roots.append(canonical_data_root(ctx))
    except Exception:
        roots.append(getattr(ctx, "budget_drive_root", None))
    roots.append(fallback_root)
    return read_verification_receipts_from_roots(roots, task_id, gap_reasons=gap_reasons)


def task_verification_receipts(
    ctx: Any, env_root: Any, task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve one worker task's local/canonical receipt view."""

    task_id = str(task.get("id") or "")
    if ctx is not None:
        return read_context_verification_receipts(
            ctx, task_id, fallback_root=task.get("budget_drive_root") or env_root,
        )
    return read_verification_receipts_from_roots(
        [env_root, task.get("budget_drive_root")], task_id,
    )
