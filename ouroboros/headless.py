"""Headless task helpers for CLI/workspace runs.

The gateway owns task transport; this module owns the small amount of local
filesystem state needed for isolated external runs and patch artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.post_task_checkpoint import project_replica_task_result_fields
from ouroboros.task_results import (
    cancellation_blocks_child_result, load_task_result, validate_task_id, write_task_result,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


HEADLESS_TASKS_DIR = pathlib.Path("state") / "headless_tasks"
ARTIFACTS_DIR = pathlib.Path("task_results") / "artifacts"
TASK_DRIVES_DIR = pathlib.Path("task_drives")
ARTIFACT_STATUS_PENDING = "pending"
ARTIFACT_STATUS_FINALIZING = "finalizing"
ARTIFACT_STATUS_READY = "ready"
ARTIFACT_STATUS_READY_WITH_CHANGES = "ready_with_changes"
ARTIFACT_STATUS_READY_NO_CHANGES = "ready_no_changes"
ARTIFACT_STATUS_MISSING = "missing"
ARTIFACT_STATUS_FAILED = "failed"

ARTIFACT_TERMINAL_STATUSES = {
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_MISSING,
    ARTIFACT_STATUS_FAILED,
}

# Mirrors task_status.SETTLED_STATUSES; a module-level import would close the
# headless → task_status → outcomes → headless cycle, and the smoke test below
# pins equality so the literal cannot drift from the SSOT.
_FINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected_duplicate"})

# Mirrors tool_capabilities.LOCAL_READONLY_SUBAGENT_MODE; a module-level import would risk
# an import cycle (same rationale as _FINAL_STATUSES above), and the smoke test pins equality
# so the literal cannot drift from this SSOT — the kind of re-derivation drift that stranded
# the reaper's artifact finalization before task_is_readonly_subagent consolidated the gate.
_LOCAL_READONLY_SUBAGENT_MODE = "local_readonly_subagent"
_ARTIFACT_LIFECYCLE_FIELDS = {
    "artifact_status",
    "artifact_error",
    "artifact_bundle",
    "artifact_finalized_at",
}
# The PURE patch/snapshot eligibility rules (env/cache dirs, junk artifacts,
# incidental lockfiles, credential-shaped names) live in their own module (size
# gate); re-exported here (same objects) because project_sources, coop_checkpoint
# and the tests address them on THIS surface. The I/O checks and the combined
# `untracked_capture_veto_reason` predicate stay below, beside the git helpers.
from ouroboros.workspace_patch_rules import (  # noqa: F401
    _ANY_SEGMENT_EXCLUDE_DIRS,
    _LOCKFILE_MANIFESTS,
    _PATCH_EXCLUDE_RULES_VERSION,
    _PATCH_JUNK_RE,
    _PATCH_MAX_UNTRACKED_FILE_BYTES,
    _SENSITIVE_EXAMPLE_SUFFIXES,
    _SENSITIVE_FILENAMES,
    _SENSITIVE_KEY_NAMES,
    _TOP_LEVEL_EXCLUDE_DIRS,
    _incidental_lockfile_excludes,
    _lockfile_manifest_for,
    _patch_exclude_reason,
    _sensitive_untracked_reason,
)

# v6.52.2: the task-scoped manifest of {ABSOLUTE_path: sha256} fingerprints the agent declared via
# run_command/run_script `scratch=[...]` (ephemeral verification files). The patch capture below
# EXCLUDES a matching untracked path ONLY while its current content still matches the recorded sha
# (so a later real file at the same path is not dropped). SSOT for the name; ouroboros.artifacts
# imports this (headless is the lower-level module).
SCRATCH_MANIFEST_NAME = ".scratch_manifest.json"
_GIT_UNBORN_HEAD = "(unborn)"


def task_state_dir(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / HEADLESS_TASKS_DIR / validate_task_id(task_id)


def task_artifacts_dir(drive_root: pathlib.Path, task_id: str, *, create: bool = True) -> pathlib.Path:
    path = pathlib.Path(drive_root) / ARTIFACTS_DIR / validate_task_id(task_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_workspace_preflight_artifact(
    parent_drive_root: pathlib.Path,
    task_id: str,
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist the full workspace preflight report as a task artifact."""

    artifact_dir = task_artifacts_dir(parent_drive_root, task_id)
    path = artifact_dir / "workspace_preflight.json"
    atomic_write_json(path, preflight, trailing_newline=True)
    raw = path.read_bytes() if path.exists() else b""
    return {
        "kind": "workspace_preflight",
        "name": "workspace_preflight.json",
        "path": str(path),
        "size": len(raw),
        "sha256": sha256(raw).hexdigest() if raw else "",
        "workspace_root": str(preflight.get("workspace_root") or ""),
    }


def prepare_task_drive(parent_drive_root: pathlib.Path, task_id: str, memory_mode: str,
                       project_id: str = "") -> Optional[pathlib.Path]:
    """Create an isolated child drive for external runs.

    ``forked`` copies stable identity/world/registry context (and, for non-project
    tasks, the global knowledge tree). ``empty`` starts with a blank data root that
    ``Memory.ensure_files`` will initialize. Any other value keeps the parent drive
    shared and returns ``None``. A project-scoped task (``project_id`` set, Phase 3b)
    is NOT seeded with the global knowledge tree — it uses the per-project store —
    so its forked child stays isolated from ``memory/knowledge``.
    """

    mode = str(memory_mode or "shared").strip().lower()
    if mode not in {"forked", "empty"}:
        return None

    task_id = validate_task_id(task_id)
    parent = pathlib.Path(parent_drive_root)
    child = task_state_dir(parent, task_id) / "data"
    child.mkdir(parents=True, exist_ok=True)
    for rel in ("memory", "logs", "state", "task_results"):
        (child / rel).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        child / "state" / "state.json",
        {
            "schema_version": 1,
            "headless_task_id": str(task_id),
            "memory_mode": mode,
            "created_at": utc_now_iso(),
        },
        trailing_newline=True,
    )
    if mode == "forked":
        _copy_stable_memory(parent, child, project_id=str(project_id or "").strip())
    return child


def _resolve_retention_days(retention_days: Optional[int]) -> int:
    """Unified GC retention for terminal task drives (see ouroboros/retention.py).
    Explicit ``retention_days`` (tests/special cases) is honored as-is and bypasses
    the owner knob; ``age_cutoff`` floors at 0, so an explicit 0 prunes everything
    before ``now`` (uniform with the worktree/service prunes). Only the default
    (None) path reads the clamped owner knob."""
    from ouroboros.retention import get_gc_retention_days

    if retention_days is None:
        return get_gc_retention_days()
    return retention_days


def _timestamp_from_result(result: Dict[str, Any], fallback: float) -> float:
    for key in ("artifact_finalized_at", "completed_at", "finished_at", "ts"):
        raw = str(result.get(key) or "").strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return float(parsed.timestamp())
        except ValueError:
            continue
    return fallback



def _live_unpromoted_child_refs(
    promotion: Any, expected_child: pathlib.Path,
) -> List[Dict[str, Any]]:
    if not isinstance(promotion, dict):
        return []
    try:
        version = int(promotion.get("schema_version") or 0)
    except (TypeError, ValueError):
        return []
    if version < 1:
        return []
    if str(promotion.get("status") or "") == "complete":
        return []
    child_root = expected_child.resolve(strict=False)
    live: List[Dict[str, Any]] = []
    for row in promotion.get("pending_refs") or []:
        if not isinstance(row, dict) or not row.get("path"):
            continue
        try:
            path = pathlib.Path(str(row["path"])).resolve(strict=False)
            path.relative_to(child_root)
        except (OSError, ValueError):
            continue
        if path.is_file():
            live.append(dict(row))
    return live


def prune_headless_task_drives(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Best-effort startup prune for copied-back terminal child drives."""

    from ouroboros.retention import age_cutoff

    parent = pathlib.Path(parent_drive_root)
    base = parent / HEADLESS_TASKS_DIR
    days = _resolve_retention_days(retention_days)
    cutoff = age_cutoff(days, now)
    report: Dict[str, Any] = {
        "retention_days": days,
        "scanned": 0,
        "pruned": [],
        "skipped": [],
        "errors": [],
        "promotion_retry": {},
    }
    if not base.is_dir():
        return report
    from ouroboros.observability import retry_pending_child_ref_promotions

    report["promotion_retry"] = retry_pending_child_ref_promotions(parent)
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        report["scanned"] += 1
        try:
            validate_task_id(task_id)
            dir_mtime = task_dir.stat().st_mtime
            try:
                from ouroboros.task_status import load_effective_task_result

                result = load_effective_task_result(parent, task_id) or {}
            except Exception:
                result = load_task_result(parent, task_id) or {}
            status = str(result.get("status") or "").lower()
            if status not in _FINAL_STATUSES:
                report["skipped"].append({"task_id": task_id, "reason": "parent_not_terminal", "status": status})
                continue
            artifact_status = str(result.get("artifact_status") or "").lower()
            if artifact_status and artifact_status not in ARTIFACT_TERMINAL_STATUSES:
                report["skipped"].append({"task_id": task_id, "reason": "artifacts_not_terminal", "artifact_status": artifact_status})
                continue
            retention_ts = _timestamp_from_result(result, dir_mtime)
            if retention_ts > cutoff:
                report["skipped"].append({"task_id": task_id, "reason": "younger_than_retention"})
                continue
            expected_child = str((task_dir / "data").resolve(strict=False))
            known_child = str(
                result.get("child_drive_root")
                or result.get("headless_child_drive_root")
                or result.get("drive_root")
                or ""
            ).strip()
            if known_child and str(pathlib.Path(known_child).resolve(strict=False)) != expected_child:
                report["skipped"].append({"task_id": task_id, "reason": "child_drive_mismatch"})
                continue
            live_unpromoted = _live_unpromoted_child_refs(
                result.get("child_ref_promotion"), pathlib.Path(expected_child),
            )
            if live_unpromoted:
                report["skipped"].append({
                    "task_id": task_id,
                    "reason": "child_refs_unpromoted",
                    "pending_ref_count": len(live_unpromoted),
                })
                continue
            shutil.rmtree(task_dir)
            report["pruned"].append({"task_id": task_id, "path": str(task_dir)})
        except Exception as exc:
            report["errors"].append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    return report


def prune_task_drives(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Best-effort startup prune for direct-task scratch drives."""

    from ouroboros.retention import age_cutoff

    parent = pathlib.Path(parent_drive_root)
    base = parent / TASK_DRIVES_DIR
    days = _resolve_retention_days(retention_days)
    cutoff = age_cutoff(days, now)
    report: Dict[str, Any] = {"retention_days": days, "scanned": 0, "pruned": [], "skipped": [], "errors": []}
    if not base.is_dir():
        return report
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        report["scanned"] += 1
        try:
            validate_task_id(task_id)
            dir_mtime = task_dir.stat().st_mtime
            try:
                from ouroboros.task_status import load_effective_task_result

                result = load_effective_task_result(parent, task_id) or {}
            except Exception:
                result = load_task_result(parent, task_id) or {}
            status = str(result.get("status") or "").lower()
            if status not in _FINAL_STATUSES:
                report["skipped"].append({"task_id": task_id, "reason": "task_not_terminal", "status": status})
                continue
            if _timestamp_from_result(result, dir_mtime) > cutoff:
                report["skipped"].append({"task_id": task_id, "reason": "younger_than_retention"})
                continue
            shutil.rmtree(task_dir)
            report["pruned"].append({"task_id": task_id, "path": str(task_dir)})
        except Exception as exc:
            report["errors"].append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    return report


def prune_task_trees(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Best-effort startup prune for ephemeral task-tree coordination ledgers
    (``data/task_trees/<root_task_id>/blackboard.jsonl``). A tree's ledger is removed once
    its ROOT task is terminal (or has no surviving result) and older than the GC retention
    window — swarm-run coordination is transient, distinct from durable project memory."""

    from ouroboros.retention import age_cutoff

    parent = pathlib.Path(parent_drive_root)
    base = parent / "task_trees"
    days = _resolve_retention_days(retention_days)
    cutoff = age_cutoff(days, now)
    report: Dict[str, Any] = {"retention_days": days, "scanned": 0, "pruned": [], "skipped": [], "errors": []}
    if not base.is_dir():
        return report
    for tree_dir in sorted(base.iterdir()):
        if not tree_dir.is_dir():
            continue
        root_id = tree_dir.name
        report["scanned"] += 1
        try:
            dir_mtime = tree_dir.stat().st_mtime
            try:
                from ouroboros.task_status import load_effective_task_result

                result = load_effective_task_result(parent, root_id) or {}
            except Exception:
                result = load_task_result(parent, root_id) or {}
            status = str(result.get("status") or "").lower()
            if status and status not in _FINAL_STATUSES:
                report["skipped"].append({"root_task_id": root_id, "reason": "root_not_terminal", "status": status})
                continue
            if _timestamp_from_result(result, dir_mtime) > cutoff:
                report["skipped"].append({"root_task_id": root_id, "reason": "younger_than_retention"})
                continue
            shutil.rmtree(tree_dir)
            report["pruned"].append({"root_task_id": root_id, "path": str(tree_dir)})
        except Exception as exc:
            report["errors"].append({"root_task_id": root_id, "error": f"{type(exc).__name__}: {exc}"})
    return report


def remove_subagent_task_drive(parent_drive_root: pathlib.Path, task_id: str) -> bool:
    """Immediately remove a subagent's child drive (used on cancel/timeout).

    Completion wins (phase A, owner 4=A): callers run this only AFTER the settled
    publication (result copied back, salvage preserved on the canonical drive), so
    removal drops bounded scratch, never a kept answer. Returns whether it removed.
    """
    parent = pathlib.Path(parent_drive_root)
    try:
        validate_task_id(task_id)
    except Exception:
        return False
    headless_base = parent / HEADLESS_TASKS_DIR / task_id
    task_drive_base = parent / TASK_DRIVES_DIR / task_id
    try:
        result = load_task_result(parent, task_id) or {}
        promotion = result.get("child_ref_promotion")
        if (
            _live_unpromoted_child_refs(promotion, headless_base / "data")
            or _live_unpromoted_child_refs(promotion, task_drive_base)
        ):
            return False
    except Exception:
        # Legacy/no-metadata cleanup behavior remains fail-soft. Newly produced
        # valid metadata is parsed by the helper without raising.
        pass
    bases = (headless_base, task_drive_base)
    removed = False
    for base in bases:
        try:
            if base.is_dir():
                shutil.rmtree(base)
                removed = True
        except Exception:
            log.debug("Failed to remove subagent task drive %s", base, exc_info=True)
    return removed


def copy_child_task_result(parent_drive_root: pathlib.Path, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy a child-drive task result back to the parent data root."""

    task_id = str(task.get("id") or "")
    if not task_id:
        return None
    canonical_existing = load_task_result(parent_drive_root, task_id) or {}
    # Cancellation is authoritative before any child-root read or artifact copy.
    if cancellation_blocks_child_result(canonical_existing):
        return canonical_existing
    child_drive = _child_drive_from_task(task)
    if child_drive is None:
        return None
    child_result = load_task_result(child_drive, task_id)
    if not isinstance(child_result, dict):
        return None
    # Canonical terminal results must not retain child-local forensic/source
    # pointers that the startup GC is about to delete. Promotion happens before
    # the one atomic canonical result write; a live ref whose destination write
    # was interrupted remains named in metadata so GC can retry safely.
    from ouroboros.observability import promote_child_task_refs

    child_result, ref_promotion = promote_child_task_refs(
        pathlib.Path(parent_drive_root), child_drive, task_id, child_result,
    )
    # W2 receipt-level handoff: the child's durable verify_and_record receipts ride
    # the SAME finalization copy-back as its artifacts (fail-soft, never blocks).
    _publish_child_verification_receipts(parent_drive_root, task_id, child_drive)
    workspace_task = (
        _workspace_root_from_task(task) is not None and not task_is_readonly_subagent(task)
    )
    child_status = str(child_result.get("status") or "completed")
    existing = canonical_existing if workspace_task and child_status in _FINAL_STATUSES else {}
    existing_artifact_status = str((existing or {}).get("artifact_status") or "").strip().lower()
    preserve_parent_artifacts = existing_artifact_status in {
        ARTIFACT_STATUS_PENDING,
        ARTIFACT_STATUS_FINALIZING,
        *ARTIFACT_TERMINAL_STATUSES,
    }
    payload = {
        key: value
        for key, value in child_result.items()
        if key not in {"task_id", "status"}
    }
    payload["child_ref_promotion"] = ref_promotion
    if isinstance(payload.get("artifacts"), list):
        payload["artifacts"] = _copy_child_artifacts_to_parent(
            parent_drive_root,
            task_id,
            child_drive,
            [item for item in payload.get("artifacts") or [] if isinstance(item, dict)],
        )
        try:
            from ouroboros.outcomes import artifact_bundle_from_result

            payload["artifact_bundle"] = artifact_bundle_from_result(payload)
        except Exception:
            payload.pop("artifact_bundle", None)
    if preserve_parent_artifacts:
        payload["artifacts"] = _merge_artifacts(
            list((existing or {}).get("artifacts") or []),
            list(payload.get("artifacts") or []),
        )
        for key in _ARTIFACT_LIFECYCLE_FIELDS:
            if key in (existing or {}):
                payload[key] = (existing or {}).get(key)
    payload.setdefault("headless_child_drive_root", str(child_drive))
    if workspace_task and child_status in _FINAL_STATUSES:
        if not preserve_parent_artifacts and existing_artifact_status not in ARTIFACT_TERMINAL_STATUSES:
            payload["artifact_status"] = ARTIFACT_STATUS_FINALIZING
        payload["child_status"] = child_status
    return write_task_result(
        parent_drive_root,
        task_id,
        child_status,
        _field_projector=project_replica_task_result_fields,
        **payload,
    )


def _publish_child_verification_receipts(
    parent_drive_root: pathlib.Path, task_id: str, child_drive: pathlib.Path
) -> None:
    """Publish the child's durable ``verification_receipts.jsonl`` to the canonical root.

    Ordinary ``verify_and_record`` receipts live under the CHILD's isolated
    drive, while actor-first ``delegation_zero_run`` is canonical immediately.
    Publish the union so a whole-file refresh cannot erase that canonical-only
    lifecycle fact. Exact-row de-duplication makes task_done + reaper/cancel
    re-entry idempotent. Fail-soft: logged, never blocks finalization."""
    try:
        from ouroboros.outcome_receipt_store import publish_verification_receipt_union

        publish_verification_receipt_union(
            parent_drive_root, task_id, child_drive,
        )
    except Exception:
        log.warning("Failed to publish child receipts for task %s", task_id, exc_info=True)


def _copy_child_artifacts_to_parent(
    parent_drive_root: pathlib.Path,
    task_id: str,
    child_drive: pathlib.Path,
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebase child-drive artifact files into the parent task artifact store."""

    from ouroboros.outcome_receipt_store import is_verification_receipts_path

    parent_dir = task_artifacts_dir(parent_drive_root, task_id)
    rebased: List[Dict[str, Any]] = []
    for artifact in artifacts:
        item = dict(artifact)
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            rebased.append(item)
            continue
        src = pathlib.Path(raw_path)
        if not src.is_absolute():
            src = (child_drive / raw_path).resolve(strict=False)
        if is_verification_receipts_path(child_drive, task_id, src):
            # ``copy_child_task_result`` publishes this stream through the
            # locked union immediately before rebasing ordinary artifacts.
            # A stale generic copy would create a competing replica (or, on a
            # reused destination, erase canonical-only lifecycle rows).
            continue
        try:
            src.resolve(strict=False).relative_to(parent_dir.resolve(strict=False))
            rebased.append(item)
            continue
        except ValueError:
            pass
        if not src.is_file():
            # The artifact path is relative/outside the child drive and the file is
            # not present, so it cannot be rebased into the parent store. Surface the
            # failure (flag + warn) instead of silently keeping an unreachable path
            # that the parent UI/consumers cannot serve.
            log.warning(
                "Child artifact for task %s could not be rebased into the parent store: %r",
                task_id, raw_path,
            )
            item["copy_status"] = "failed"
            item["copy_error"] = "artifact file not found for rebase"
            rebased.append(item)
            continue
        dest = parent_dir / src.name
        if dest.exists() and dest.resolve(strict=False) != src.resolve(strict=False):
            dest = parent_dir / f"{src.stem}_{sha256(str(src).encode('utf-8')).hexdigest()[:8]}{src.suffix}"
        shutil.copy2(src, dest)
        data = dest.read_bytes()
        item["path"] = str(dest)
        item["name"] = str(item.get("name") or dest.name)
        item["size"] = len(data)
        item["sha256"] = sha256(data).hexdigest()
        rebased.append(item)
    return rebased


def task_is_readonly_subagent(task: Dict[str, Any]) -> bool:
    """A local-readonly live subagent produces no durable owner-facing artifacts, so the
    ``task_done`` finalize path (and the reaper that honors a self-finalized result) skip
    artifact finalization for it. Single SSOT gate so every call site reads the same rule
    instead of re-deriving it (a re-derivation drift is what stranded the reaper path)."""
    if not isinstance(task, dict):
        return False
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    task_constraint = task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {}
    if not task_constraint and isinstance(metadata.get("task_constraint"), dict):
        task_constraint = metadata.get("task_constraint") or {}
    return (
        str(task.get("delegation_role") or metadata.get("delegation_role") or "") == "subagent"
        and str(task_constraint.get("mode") or "") == _LOCAL_READONLY_SUBAGENT_MODE
    )


_DELIVERABLE_MANIFEST_FILE_CAP = 10000
_DELIVERABLE_MANIFEST_HASH_CHUNK = 1024 * 1024  # 1 MiB streaming chunks (bounded memory)
# Files larger than this are recorded by size only (hash skipped) so a single huge
# binary/media/build artifact cannot wedge or OOM genesis finalization.
_DELIVERABLE_MANIFEST_HASH_BYTE_CAP = 64 * 1024 * 1024  # 64 MiB


def _build_deliverable_manifest(
    workspace_root: pathlib.Path, task_id: str, project_id: str
) -> Dict[str, Any]:
    """Typed content listing of a from-scratch (genesis) project's deliverables
    (deferral 3): rel path + size + sha256 per file, surfaced on the artifact axis so a
    genesis project's OUTPUT (not just its patch diff) is inspectable. Excludes VCS and
    virtualenv junk. P1 fail-loud: if the tree exceeds the file cap, ``truncated`` is set
    instead of silently dropping files. Hashing STREAMS in fixed chunks (never loads a
    whole file into memory) and skips the hash for files over the byte cap, so a large
    artifact can neither OOM nor wedge finalization."""
    import hashlib

    contents: List[Dict[str, Any]] = []
    count = 0
    truncated = False
    for root, dirs, files in os.walk(workspace_root):
        dirs[:] = [d for d in dirs if d not in _TOP_LEVEL_EXCLUDE_DIRS and d != ".git"]
        for fname in sorted(files):
            if count >= _DELIVERABLE_MANIFEST_FILE_CAP:
                truncated = True
                break
            fpath = pathlib.Path(root) / fname
            if fpath.is_symlink():
                # SECURITY: never follow a symlink out of the project — a genesis child
                # could point one at an owner/runtime file outside workspace_root, and
                # stat()/open() would then read/hash bytes outside the deliverable tree.
                # Record it as a symlink WITHOUT reading the target.
                contents.append({
                    "rel": str(fpath.relative_to(workspace_root)),
                    "symlink": True,
                    "sha256": "",
                })
                count += 1
                continue
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            entry: Dict[str, Any] = {"rel": str(fpath.relative_to(workspace_root)), "size": size}
            if size > _DELIVERABLE_MANIFEST_HASH_BYTE_CAP:
                entry["sha256"] = ""
                entry["hash_skipped"] = "size_over_cap"
            else:
                try:
                    h = hashlib.sha256()
                    with open(fpath, "rb") as fh:
                        for chunk in iter(lambda: fh.read(_DELIVERABLE_MANIFEST_HASH_CHUNK), b""):
                            h.update(chunk)
                    entry["sha256"] = h.hexdigest()
                except Exception:
                    continue
            contents.append(entry)
            count += 1
        if truncated:
            break
    return {
        "schema_version": 1,
        "task_id": task_id,
        "project_id": project_id,
        "project_root": str(workspace_root),
        "created_at": utc_now_iso(),
        "file_count": count,
        "truncated": truncated,
        "contents": contents,
    }


def finalize_task_artifacts(parent_drive_root: pathlib.Path, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Write patch/memory-export artifacts for a completed headless task."""

    artifacts: List[Dict[str, Any]] = []
    task_id = str(task.get("id") or "")
    if not task_id:
        return artifacts

    existing = load_task_result(parent_drive_root, task_id) or {}
    # A cancellation latch wins before artifact creation or surviving-root reads.
    if cancellation_blocks_child_result(existing):
        return artifacts
    artifact_dir = task_artifacts_dir(parent_drive_root, task_id)
    workspace_root = _workspace_root_from_task(task)
    status = str(existing.get("status") or "completed")
    artifact_status = ARTIFACT_STATUS_READY
    artifact_error = ""
    if workspace_root is not None:
        write_task_result(
            parent_drive_root,
            task_id,
            status,
            artifact_status=ARTIFACT_STATUS_FINALIZING,
        )
        try:
            patch_artifacts, manifest = write_workspace_patch_artifacts(
                workspace_root,
                artifact_dir,
                task=task,
            )
            artifacts.extend(patch_artifacts)
            artifact_status = str(manifest.get("status") or ARTIFACT_STATUS_READY_WITH_CHANGES)
            if manifest.get("status") == ARTIFACT_STATUS_FAILED:
                artifact_status = ARTIFACT_STATUS_FAILED
                artifact_error = "; ".join(str(err.get("message") or err) for err in manifest.get("errors") or [])[:1000]
        except Exception as exc:
            artifact_status = ARTIFACT_STATUS_FAILED
            artifact_error = f"{type(exc).__name__}: {exc}"
            manifest_path = artifact_dir / "workspace_patch.json"
            manifest = _empty_patch_manifest(
                workspace_root,
                status=ARTIFACT_STATUS_FAILED,
                errors=[{"type": "exception", "message": artifact_error}],
            )
            atomic_write_json(
                manifest_path,
                manifest,
                trailing_newline=True,
            )
            artifacts.append({
                "kind": "workspace_patch_manifest",
                "name": "workspace_patch.json",
                "path": str(manifest_path),
                "size": manifest_path.stat().st_size if manifest_path.exists() else 0,
                "workspace_root": str(workspace_root),
            })

    child_drive = _child_drive_from_task(task)
    if child_drive is not None:
        try:
            export_path = artifact_dir / "memory_export.json"
            atomic_write_json(export_path, build_memory_export(child_drive, task), trailing_newline=True)
            artifacts.append({
                "kind": "memory_export",
                "name": "memory_export.json",
                "path": str(export_path),
                "size": export_path.stat().st_size if export_path.exists() else 0,
                "memory_mode": str(task.get("memory_mode") or ""),
            })
        except Exception as exc:
            if workspace_root is not None:
                artifact_status = ARTIFACT_STATUS_FAILED
            message = f"{type(exc).__name__}: {exc}"
            artifact_error = f"{artifact_error}; {message}" if artifact_error else message

    # Deferral 3: a from-scratch (genesis) project gets a typed deliverable manifest on
    # the artifact axis, so its OUTPUT files (not only the patch diff) are inspectable.
    tc = task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else \
        (existing.get("task_constraint") if isinstance(existing.get("task_constraint"), dict) else {})
    if (
        workspace_root is not None
        and str((tc or {}).get("surface") or "") == "genesis"
        and workspace_root.is_dir()
    ):
        try:
            manifest_path = artifact_dir / "deliverable_manifest.json"
            dm = _build_deliverable_manifest(workspace_root, task_id, str(task.get("project_id") or ""))
            atomic_write_json(manifest_path, dm, trailing_newline=True)
            artifacts.append({
                "kind": "deliverable_manifest",
                "name": "deliverable_manifest.json",
                "path": str(manifest_path),
                "size": manifest_path.stat().st_size if manifest_path.exists() else 0,
                "file_count": int(dm.get("file_count") or 0),
                "truncated": bool(dm.get("truncated")),
                "workspace_root": str(workspace_root),
            })
            if dm.get("truncated"):
                log.warning(
                    "deliverable_manifest truncated at cap %d for task %s",
                    _DELIVERABLE_MANIFEST_FILE_CAP, task_id,
                )
        except Exception as exc:
            log.debug("deliverable_manifest build failed for %s: %s", task_id, exc, exc_info=True)

    if artifacts or workspace_root is not None:
        existing = load_task_result(parent_drive_root, task_id) or {}
        drop_kinds = {"workspace_patch"} if workspace_root is not None and artifact_status == ARTIFACT_STATUS_FAILED else set()
        merged = _merge_artifacts(list(existing.get("artifacts") or []), artifacts, drop_kinds=drop_kinds)
        fields: Dict[str, Any] = {
            "artifacts": merged,
            "artifact_status": artifact_status if workspace_root is not None else str(existing.get("artifact_status") or ""),
            "artifact_finalized_at": utc_now_iso(),
        }
        if artifact_error:
            fields["artifact_error"] = artifact_error
        # ``fields`` already carries "artifacts" and "artifact_status".
        provisional = {**existing, **fields}
        provisional.pop("artifact_bundle", None)
        try:
            from ouroboros.outcomes import artifact_bundle_from_result, refresh_verification_ledger_artifacts

            artifact_bundle = artifact_bundle_from_result(provisional)
            fields["artifact_bundle"] = artifact_bundle
            axes = existing.get("outcome_axes") if isinstance(existing.get("outcome_axes"), dict) else {}
            if axes:
                axes = dict(axes)
                artifact_axis = dict(axes.get("artifacts") or {})
                artifact_axis["status"] = str(artifact_bundle.get("status") or artifact_status or "")
                axes["artifacts"] = artifact_axis
                fields["outcome_axes"] = axes
            refreshed_ledger = refresh_verification_ledger_artifacts(
                existing.get("verification_ledger"),
                artifact_bundle,
            )
            if refreshed_ledger is not None:
                fields["verification_ledger"] = refreshed_ledger
            for item in merged:
                if not isinstance(item, dict) or str(item.get("kind") or "") != "verification_ledger":
                    continue
                ledger_path = pathlib.Path(str(item.get("path") or ""))
                if not ledger_path.is_file():
                    continue
                try:
                    raw_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                    refreshed_artifact_ledger = refresh_verification_ledger_artifacts(raw_ledger, artifact_bundle)
                    if isinstance(refreshed_artifact_ledger, dict):
                        atomic_write_json(ledger_path, refreshed_artifact_ledger, trailing_newline=True)
                        data = ledger_path.read_bytes()
                        item["size"] = len(data)
                        item["sha256"] = sha256(data).hexdigest()
                        item["status"] = ARTIFACT_STATUS_READY
                except Exception:
                    log.debug("Failed to refresh verification ledger artifact for task %s", task_id, exc_info=True)
        except Exception:
            pass
        write_task_result(
            parent_drive_root,
            task_id,
            str(existing.get("status") or status or "completed"),
            **fields,
        )
    return artifacts


def build_workspace_patch(workspace_root: pathlib.Path) -> str:
    """Return a git patch for tracked changes plus untracked files."""

    with tempfile.TemporaryDirectory() as tmp:
        artifacts, manifest = write_workspace_patch_artifacts(
            pathlib.Path(workspace_root),
            pathlib.Path(tmp),
            task={},
        )
        if manifest.get("status") == ARTIFACT_STATUS_FAILED:
            return ""
        for artifact in artifacts:
            if artifact.get("kind") == "workspace_patch":
                path = pathlib.Path(str(artifact.get("path") or ""))
                return path.read_text(encoding="utf-8") if path.is_file() else ""
    return ""


def write_workspace_patch_artifacts(
    workspace_root: pathlib.Path,
    artifact_dir: pathlib.Path,
    *,
    task: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stream workspace patch and manifest artifacts into ``artifact_dir``."""

    root = pathlib.Path(workspace_root).resolve(strict=False)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    patch_path = artifact_dir / "workspace.patch"
    manifest_path = artifact_dir / "workspace_patch.json"
    errors: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    tracked_excluded: List[Dict[str, str]] = []
    sensitive: List[Dict[str, str]] = []
    included_untracked: List[str] = []
    acting_constraint = _acting_constraint_from_task(task)
    task_base_sha = str(acting_constraint.base_sha or "").strip() if acting_constraint else ""
    preflight_head = _preflight_head_from_task(task)
    if not task_base_sha and not preflight_head and _preflight_head_present(task):
        preflight_head = _GIT_UNBORN_HEAD
    base_ref, base_head, base_is_empty_tree = _workspace_patch_base(
        root,
        errors,
        expected_base_sha=task_base_sha or preflight_head,
    )
    changed_tracked = _git_path_list(
        ["git", "diff", "--name-only", "-z", "--no-ext-diff", "--no-color", base_ref, "--"],
        root,
        errors,
    )
    diffstat = ""
    untracked = _git_path_list(["git", "ls-files", "-z", "--others", "--exclude-standard"], root, errors)
    # v6.52.2: exclude declared ephemeral scratch (run_command/run_script `scratch=[...]`) so a
    # throwaway verification file the agent forgot to delete never leaks into the workspace patch.
    # The manifest stores {abs_path: sha256}; a file is excluded ONLY while its CURRENT content
    # still matches the recorded scratch sha — so a LATER real file written to the same path
    # (different content) is NOT dropped. Empty/absent/mismatched => included (no regression).
    scratch_sha_by_rel: dict = {}
    scratch_sha_by_abs: dict = {}
    try:
        _scratch_map = json.loads((artifact_dir / SCRATCH_MANIFEST_NAME).read_text(encoding="utf-8")).get("scratch")
        if isinstance(_scratch_map, dict):
            for _abs, _sha in _scratch_map.items():
                try:
                    _resolved = pathlib.Path(str(_abs)).resolve(strict=False)
                    scratch_sha_by_abs[os.path.normcase(str(_resolved))] = str(_sha)
                    scratch_sha_by_rel[_resolved.relative_to(root).as_posix()] = str(_sha)
                except Exception:
                    continue
    except Exception:
        scratch_sha_by_rel = {}
        scratch_sha_by_abs = {}
    for rel in untracked:
        _want_sha = scratch_sha_by_rel.get(rel) or scratch_sha_by_abs.get(os.path.normcase(str((root / rel).resolve(strict=False))))
        if _want_sha:
            try:
                _cur_sha = sha256((root / rel).read_bytes()).hexdigest()
            except OSError:
                _cur_sha = None
            if _cur_sha == _want_sha:
                excluded.append({"path": rel, "reason": "declared ephemeral scratch (v6.52.2)"})
                continue
        sensitive_reason = _sensitive_untracked_reason(rel)
        if sensitive_reason:
            sensitive.append({"path": rel, "reason": sensitive_reason})
            continue
        reason = _patch_exclude_reason(rel)
        if reason:
            excluded.append({"path": rel, "reason": reason})
            continue
        blob_reason = _untracked_blob_exclude_reason(root, rel)
        if blob_reason:
            excluded.append({"path": rel, "reason": blob_reason})
            continue
        included_untracked.append(rel)
    incidental_lock_excludes = _incidental_lockfile_excludes([*changed_tracked, *included_untracked])
    if incidental_lock_excludes:
        kept_untracked: List[str] = []
        for rel in included_untracked:
            if rel in incidental_lock_excludes:
                excluded.append({"path": rel, "reason": "incidental lockfile without sibling manifest change"})
            else:
                kept_untracked.append(rel)
        included_untracked = kept_untracked
    # A sensitive-shaped untracked file is a PER-FILE exclusion (disclosed in
    # ``sensitive_blocked``), never a manifest error: the error path suppressed
    # the ENTIRE patch write, so one credential-shaped NAME (public.pem,
    # token_count.json) annihilated the whole tracked diff. The snapshot path
    # (untracked_capture_veto_reason → provision_execution_snapshot) always
    # treated the same hit as a per-file skip; the patch now matches it (#447).

    hasher = sha256()
    total_size = 0
    with patch_path.open("wb") as fh:
        if not errors:
            tracked_lock_excludes = sorted(set(changed_tracked) & incidental_lock_excludes)
            tracked_pathspec = ["--"]
            if tracked_lock_excludes:
                tracked_pathspec += ["."] + [f":(exclude){rel}" for rel in tracked_lock_excludes]
                for rel in tracked_lock_excludes:
                    tracked_excluded.append({"path": rel, "reason": "incidental lockfile without sibling manifest change"})
            diffstat = _git_stdout(
                ["git", "diff", "--stat", "--no-ext-diff", "--no-color", base_ref, *tracked_pathspec],
                root,
                allow_rc={0},
                errors=errors,
            )
            total_size += _append_git_output(
                ["git", "diff", "--binary", "--no-ext-diff", "--no-color", base_ref, *tracked_pathspec],
                root,
                fh,
                hasher,
                allow_rc={0},
                errors=errors,
                diagnostics=diagnostics,
            )
            for rel in included_untracked:
                if total_size:
                    total_size += _write_patch_separator(fh, hasher)
                total_size += _append_git_output(
                    ["git", "diff", "--no-index", "--binary", "--no-ext-diff", "--no-color", "--", os.devnull, rel],
                    root,
                    fh,
                    hasher,
                    allow_rc={0, 1},
                    errors=errors,
                    diagnostics=diagnostics,
                )
    if errors:
        try:
            patch_path.unlink()
        except OSError:
            pass
        total_size = 0
        digest = ""
    else:
        digest = hasher.hexdigest()

    head_error: Dict[str, Any] | None = None
    head_errors: List[Dict[str, Any]] = []
    current_head = _git_stdout(["git", "rev-parse", "--verify", "HEAD"], root, allow_rc={0}, errors=head_errors).strip()
    # Q11: the moved-HEAD fail-closed tripwire applies ONLY to a child's private
    # self_worktree, where a moved HEAD can only mean the worktree itself
    # rewrote history under the patch (its base is always a real provisioned
    # commit, never unborn). In a SHARED tree (external_workspace/genesis) the
    # parent's own legitimate commits move HEAD too — enforcing it there failed
    # every innocent in-flight sibling; shared-tree integrity is verified by the
    # reverse-patch check in tools/subagent_integration (verified_shared_workspace),
    # and base_sha stays the patch BASE so parent-committed work is still captured.
    if task_base_sha and acting_constraint is not None and acting_constraint.surface == "self_worktree":
        if not current_head:
            errors.extend(head_errors)
            head_error = {
                "type": "workspace_head_unverified",
                "message": "workspace HEAD could not be verified at artifact finalization",
                "expected_head": base_head,
                "current_head": "",
            }
            errors.append(head_error)
        elif current_head != base_head:
            head_error = {
                "type": "workspace_head_changed",
                "message": "workspace HEAD changed during task execution; patch artifact is invalid",
                "expected_head": base_head,
                "current_head": current_head,
            }
            errors.append(head_error)
    if head_error:
        try:
            patch_path.unlink()
        except OSError:
            pass
        total_size = 0
        digest = ""

    if errors:
        status = ARTIFACT_STATUS_FAILED
    elif total_size > 0:
        status = ARTIFACT_STATUS_READY_WITH_CHANGES
    else:
        status = ARTIFACT_STATUS_READY_NO_CHANGES
        try:
            patch_path.unlink()
        except OSError:
            pass
        digest = ""
    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": status,
        "workspace_root": str(root),
        "patch_name": "workspace.patch",
        "manifest_name": "workspace_patch.json",
        "base_ref": base_ref,
        "base_head": base_head,
        "base_is_empty_tree": base_is_empty_tree,
        "current_head": current_head or (_GIT_UNBORN_HEAD if base_is_empty_tree else ""),
        "patch_size": total_size,
        "sha256": digest,
        "diffstat": diffstat,
        "counts": {
            "tracked_changed": len(changed_tracked),
            "tracked_excluded": len(tracked_excluded),
            "untracked_included": len(included_untracked),
            "untracked_excluded": len(excluded),
            "sensitive_blocked": len(sensitive),
        },
        "tracked_changed": changed_tracked,
        "tracked_excluded": tracked_excluded,
        "untracked_included": included_untracked,
        "untracked_excluded": excluded,
        "sensitive_blocked": sensitive,
        "exclude_rules_version": _PATCH_EXCLUDE_RULES_VERSION,
        "diagnostics": diagnostics,
        "errors": errors,
    }
    atomic_write_json(manifest_path, manifest, trailing_newline=True)
    artifacts = [
        {
            "kind": "workspace_patch_manifest",
            "name": "workspace_patch.json",
            "path": str(manifest_path),
            "size": manifest_path.stat().st_size if manifest_path.exists() else 0,
            "workspace_root": str(root),
        }
    ]
    if status == ARTIFACT_STATUS_READY_WITH_CHANGES:
        artifacts.insert(0, {
            "kind": "workspace_patch",
            "name": "workspace.patch",
            "path": str(patch_path),
            "size": total_size,
            "sha256": digest,
            "workspace_root": str(root),
        })
    return artifacts, manifest


def build_memory_export(child_drive_root: pathlib.Path, task: Dict[str, Any]) -> Dict[str, Any]:
    """Create an explicit export artifact without merging it into parent memory."""

    root = pathlib.Path(child_drive_root)
    memory_root = root / "memory"
    files: Dict[str, str] = {}
    if memory_root.is_dir():
        for path in sorted(memory_root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                rel = str(path.relative_to(memory_root)).replace(os.sep, "/")
                files[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "task_id": str(task.get("id") or ""),
        "memory_mode": str(task.get("memory_mode") or ""),
        "child_drive_root": str(root),
        "files": files,
    }


def _copy_stable_memory(parent: pathlib.Path, child: pathlib.Path, *, project_id: str = "") -> None:
    parent_memory = parent / "memory"
    child_memory = child / "memory"
    for rel in ("identity.md", "WORLD.md", "registry.md"):
        src = parent_memory / rel
        if src.is_file():
            dst = child_memory / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # Project-scoped tasks use the per-project knowledge store, so do NOT seed the
    # forked child with the global knowledge TOPICS/index (keeps it isolated from
    # memory/knowledge). identity/WORLD/registry carry for P1 continuity, and the
    # global Pattern Register (general cross-project error patterns) still carries.
    if str(project_id or "").strip():
        src_patterns = parent_memory / "knowledge" / "patterns.md"
        if src_patterns.is_file():
            dst_patterns = child_memory / "knowledge" / "patterns.md"
            dst_patterns.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_patterns, dst_patterns)
        return
    src_knowledge = parent_memory / "knowledge"
    dst_knowledge = child_memory / "knowledge"
    if src_knowledge.is_dir():
        shutil.copytree(src_knowledge, dst_knowledge, dirs_exist_ok=True)


def _child_drive_from_task(task: Dict[str, Any]) -> Optional[pathlib.Path]:
    text = str(task.get("drive_root") or task.get("child_drive_root") or "").strip()
    return pathlib.Path(text) if text else None


def _workspace_root_from_task(task: Dict[str, Any]) -> Optional[pathlib.Path]:
    text = str(task.get("workspace_root") or "").strip()
    if not text:
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        text = str(meta.get("workspace_root") or "").strip()
    return pathlib.Path(text) if text else None


def _git_stdout(
    cmd: Sequence[str],
    cwd: pathlib.Path,
    *,
    allow_rc: Iterable[int] = (0,),
    errors: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Text projection of ``_git_bytes`` (same rc/timeout/error handling)."""
    return _git_bytes(cmd, cwd, allow_rc=allow_rc, errors=errors).decode("utf-8", errors="replace")


def _workspace_patch_base(
    root: pathlib.Path,
    errors: List[Dict[str, Any]],
    *,
    expected_base_sha: str = "",
) -> Tuple[str, str, bool]:
    """Return the git tree-ish used as the patch baseline.

    A freshly initialized external workspace is a valid git worktree even when
    it has no commits. In that state ``git diff HEAD`` fails, so patch capture
    compares against Git's canonical empty tree instead of forcing adapters to
    create a synthetic target commit in the user's workspace.
    """

    if expected_base_sha:
        if expected_base_sha == _GIT_UNBORN_HEAD:
            empty_tree = _git_empty_tree_oid(root, errors)
            if empty_tree:
                return empty_tree, _GIT_UNBORN_HEAD, True
            return "HEAD", _GIT_UNBORN_HEAD, False
        if not _looks_like_git_oid(expected_base_sha):
            errors.append({
                "type": "workspace_base_sha_invalid",
                "message": "acting subagent base_sha is not a git object id; refusing to build patch artifact",
                "base_sha": expected_base_sha,
            })
            return "HEAD", expected_base_sha, False
        verify_errors: List[Dict[str, Any]] = []
        resolved = _git_stdout(
            ["git", "rev-parse", "--verify", f"{expected_base_sha}^{{commit}}"],
            root,
            allow_rc={0},
            errors=verify_errors,
        ).strip()
        if not resolved:
            errors.extend(verify_errors)
            errors.append({
                "type": "workspace_base_sha_missing",
                "message": "acting subagent base_sha is not available in workspace git history",
                "base_sha": expected_base_sha,
            })
            return expected_base_sha, expected_base_sha, False
        return resolved, resolved, False

    head_errors: List[Dict[str, Any]] = []
    head = _git_stdout(["git", "rev-parse", "--verify", "HEAD"], root, allow_rc={0}, errors=head_errors).strip()
    if head:
        return head, head, False

    worktree_errors: List[Dict[str, Any]] = []
    inside = _git_stdout(
        ["git", "rev-parse", "--is-inside-work-tree"],
        root,
        allow_rc={0},
        errors=worktree_errors,
    ).strip()
    if inside == "true" and _head_reflog_exists(root):
        errors.extend(head_errors)
        errors.append({
            "type": "git_invalid_head",
            "command": ["git", "rev-parse", "--verify", "HEAD"],
            "message": "HEAD could not be resolved but the repository has HEAD history; refusing to treat it as unborn",
        })
        return "HEAD", "", False
    if inside == "true":
        empty_tree = _git_empty_tree_oid(root, errors)
        if empty_tree:
            return empty_tree, _GIT_UNBORN_HEAD, True

    errors.extend(head_errors or worktree_errors)
    return "HEAD", "", False


def _git_empty_tree_oid(root: pathlib.Path, errors: List[Dict[str, Any]]) -> str:
    try:
        result = subprocess.run(
            ["git", "hash-object", "-t", "tree", "--stdin"],
            cwd=str(root),
            input="",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        errors.append({"type": "git_exception", "command": ["git", "hash-object", "-t", "tree", "--stdin"], "message": f"{type(exc).__name__}: {exc}"})
        return ""
    if result.returncode != 0:
        errors.append({
            "type": "git_error",
            "command": ["git", "hash-object", "-t", "tree", "--stdin"],
            "returncode": result.returncode,
            "stderr": (result.stderr or "")[-2000:],
        })
        return ""
    return (result.stdout or "").strip()


def _head_reflog_exists(root: pathlib.Path) -> bool:
    path_text = _git_stdout(["git", "rev-parse", "--git-path", "logs/HEAD"], root, allow_rc={0}).strip()
    if not path_text:
        return False
    path = pathlib.Path(path_text)
    if not path.is_absolute():
        path = root / path
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _looks_like_git_oid(value: str) -> bool:
    text = str(value or "").strip()
    return 7 <= len(text) <= 64 and all(ch in "0123456789abcdefABCDEF" for ch in text)


def _git_path_list(cmd: Sequence[str], root: pathlib.Path, errors: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    output = _git_bytes(cmd, root, errors=errors)
    if not output:
        return []
    return [part.decode("utf-8", errors="replace") for part in output.split(b"\0") if part]


def _git_bytes(
    cmd: Sequence[str],
    cwd: pathlib.Path,
    *,
    allow_rc: Iterable[int] = (0,),
    errors: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    try:
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        if errors is not None:
            errors.append({"type": "git_timeout", "command": list(cmd), "message": "git command timed out"})
        return b""
    except Exception as exc:
        if errors is not None:
            errors.append({"type": "git_exception", "command": list(cmd), "message": f"{type(exc).__name__}: {exc}"})
        return b""
    if result.returncode not in set(allow_rc):
        if errors is not None:
            errors.append({
                "type": "git_error",
                "command": list(cmd),
                "returncode": result.returncode,
                "stderr": (result.stderr or b"").decode("utf-8", errors="replace")[-2000:],
            })
        return b""
    return result.stdout or b""


def _append_git_output(
    cmd: Sequence[str],
    cwd: pathlib.Path,
    fh: BinaryIO,
    hasher: Any,
    *,
    allow_rc: set[int],
    errors: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
) -> int:
    written_box = {"value": 0}
    read_errors: List[str] = []
    try:
        with tempfile.TemporaryFile() as stderr_fh:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
            )
            assert proc.stdout is not None

            def _reader() -> None:
                try:
                    while True:
                        chunk = proc.stdout.read(1024 * 128)
                        if not chunk:
                            break
                        fh.write(chunk)
                        hasher.update(chunk)
                        written_box["value"] += len(chunk)
                except Exception as exc:
                    read_errors.append(f"{type(exc).__name__}: {exc}")

            reader = threading.Thread(target=_reader, name="workspace-patch-git-stdout", daemon=True)
            reader.start()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                reader.join(timeout=5)
                if reader.is_alive():
                    errors.append({"type": "git_timeout", "command": list(cmd), "message": "git stdout reader timed out"})
                errors.append({"type": "git_timeout", "command": list(cmd), "message": "git command timed out"})
                return int(written_box["value"])
            reader.join(timeout=5)
            if reader.is_alive():
                errors.append({"type": "git_timeout", "command": list(cmd), "message": "git stdout reader timed out"})
            for read_error in read_errors:
                errors.append({"type": "git_exception", "command": list(cmd), "message": read_error})
            stderr_fh.seek(0)
            stderr = stderr_fh.read() or b""
    except subprocess.TimeoutExpired:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        errors.append({"type": "git_timeout", "command": list(cmd), "message": "git command timed out"})
        return int(written_box["value"])
    except Exception as exc:
        errors.append({"type": "git_exception", "command": list(cmd), "message": f"{type(exc).__name__}: {exc}"})
        return int(written_box["value"])
    if proc.returncode not in allow_rc:
        errors.append({
            "type": "git_error",
            "command": list(cmd),
            "returncode": proc.returncode,
            "stderr": stderr.decode("utf-8", errors="replace")[-2000:],
        })
    written = int(written_box["value"])
    diagnostics.append({"command": list(cmd), "returncode": proc.returncode, "bytes": written})
    return written


def _write_patch_separator(fh: BinaryIO, hasher: Any) -> int:
    data = b"\n"
    fh.write(data)
    hasher.update(data)
    return len(data)


# PEM private-key header (any label: RSA/EC/DSA/OPENSSH/ENCRYPTED/none). This is
# CONTENT evidence — the filename-shape rules in workspace_patch_rules miss real
# key material under an innocent name (notes.txt) while flagging public.pem; a
# bounded head read catches the former on evidence, not on spelling (#447).
_PEM_PRIVATE_KEY_RE = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_PEM_HEAD_READ_BYTES = 4096


def _untracked_blob_exclude_reason(root: pathlib.Path, rel: str) -> str:
    """Reason to drop an untracked file from the workspace patch when it is a
    build/runtime BINARY, exceeds the per-file size cap, or carries a PEM
    private-key header in its head bytes. Keeps real-usage patches
    source-shaped without losing data (the file stays in the workspace
    and is recorded under ``untracked_excluded``). On any git/stat failure the
    file is INCLUDED (conservative — the main binary diff still applies)."""

    try:
        size = (root / rel).lstat().st_size
    except OSError:
        return ""  # unreadable/symlink races: include and let git decide
    if size > _PATCH_MAX_UNTRACKED_FILE_BYTES:
        return f"untracked file exceeds size cap ({size}B > {_PATCH_MAX_UNTRACKED_FILE_BYTES}B)"
    try:
        with (root / rel).open("rb") as fh:
            head = fh.read(_PEM_HEAD_READ_BYTES)
    except OSError:
        head = b""
    if _PEM_PRIVATE_KEY_RE.search(head):
        return "private key material (PEM private-key header)"
    numstat = _git_stdout(
        ["git", "diff", "--no-index", "--numstat", "--no-ext-diff", "--no-color", "--", os.devnull, rel],
        root,
        allow_rc={0, 1},
        errors=None,
    )
    first = numstat.strip().splitlines()[0] if numstat.strip() else ""
    if first.startswith("-\t-"):
        return "binary file"
    return ""


def untracked_capture_veto_reason(root: pathlib.Path, rel: str) -> str:
    """Why an untracked file must NOT ride into a workspace snapshot or patch.

    The delegated-run baseline snapshot
    (``subagent_worktrees.provision_execution_snapshot``) asks the SAME three
    checks, in the SAME order, that ``write_workspace_patch_artifacts`` applies
    to untracked files: sensitive/credential-shaped names first, then the
    static junk rules, then the binary/size veto. One combined predicate here so
    the snapshot and the patch cannot drift apart about eligibility.
    Returns the human-readable reason, or "" when the file is eligible.
    """
    reason = _sensitive_untracked_reason(rel)
    if reason:
        return reason
    reason = _patch_exclude_reason(rel)
    if reason:
        return reason
    return _untracked_blob_exclude_reason(root, rel)


def _preflight_head_from_task(task: Dict[str, Any]) -> str:
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    return str(git.get("head") or "")


def _preflight_head_present(task: Dict[str, Any]) -> bool:
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    return "head" in git


def _acting_constraint_from_task(task: Dict[str, Any]):
    """Normalized acting-subagent constraint carried by ``task``, or None."""
    raw = task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {}
    if not raw:
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        raw = meta.get("task_constraint") if isinstance(meta.get("task_constraint"), dict) else {}
    try:
        constraint = normalize_task_constraint(raw)
    except Exception:
        return None
    return constraint if constraint and constraint.mode == "acting_subagent" else None


def _empty_patch_manifest(
    workspace_root: pathlib.Path,
    *,
    status: str,
    errors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": status,
        "workspace_root": str(workspace_root),
        "patch_name": "workspace.patch",
        "manifest_name": "workspace_patch.json",
        "base_ref": "",
        "base_head": "",
        "base_is_empty_tree": False,
        "current_head": "",
        "patch_size": 0,
        "sha256": "",
        "diffstat": "",
        "counts": {
            "tracked_changed": 0,
            "untracked_included": 0,
            "untracked_excluded": 0,
            "sensitive_blocked": 0,
        },
        "tracked_changed": [],
        "untracked_included": [],
        "untracked_excluded": [],
        "sensitive_blocked": [],
        "exclude_rules_version": _PATCH_EXCLUDE_RULES_VERSION,
        "diagnostics": [],
        "errors": errors,
    }


def _merge_artifacts(
    existing: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
    *,
    drop_kinds: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    drop = drop_kinds or set()
    key_for = lambda item: (
        str(item.get("kind") or ""),
        str(item.get("name") or pathlib.Path(str(item.get("path") or "")).name),
    )
    keys = {key_for(item) for item in new_items if isinstance(item, dict)}
    for item in existing:
        if not isinstance(item, dict):
            continue
        key = key_for(item)
        if key[0] not in drop and key not in keys:
            merged.append(item)
    merged.extend(new_items)
    return merged


__all__ = [
    "ARTIFACT_STATUS_FAILED",
    "ARTIFACT_STATUS_FINALIZING",
    "ARTIFACT_STATUS_PENDING",
    "ARTIFACT_STATUS_READY",
    "build_memory_export",
    "build_workspace_patch",
    "copy_child_task_result",
    "finalize_task_artifacts",
    "task_is_readonly_subagent",
    "prepare_task_drive",
    "prune_headless_task_drives",
    "prune_task_drives",
    "task_artifacts_dir",
    "task_state_dir",
    "write_workspace_patch_artifacts",
    "write_workspace_preflight_artifact",
]
