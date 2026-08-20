"""Headless task helpers for CLI/workspace runs.

The gateway owns task transport; this module owns the small amount of local
filesystem state needed for isolated external runs and patch artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess  # noqa: F401
import tempfile  # noqa: F401
import threading  # noqa: F401
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

from ouroboros.contracts.task_constraint import normalize_task_constraint  # noqa: F401
from ouroboros.task_results import (
    TASK_COST_META_FIELDS,
    cancellation_blocks_child_result, load_task_result, validate_task_id, write_task_result,
)
from ouroboros.utils import atomic_write_json, replace_atomic, utc_now_iso
from ouroboros.headless_status import (  # noqa: F401
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_MISSING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    ARTIFACT_TERMINAL_STATUSES,
    _ARTIFACT_LIFECYCLE_FIELDS,
    _FINAL_STATUSES,
    _LOCAL_READONLY_SUBAGENT_MODE,
)
from ouroboros.workspace_patch_capture import (  # noqa: F401
    SCRATCH_MANIFEST_NAME,
    _GIT_UNBORN_HEAD,
    _acting_constraint_from_task,
    _append_git_output,
    _empty_patch_manifest,
    _git_bytes,
    _git_empty_tree_oid,
    _git_path_list,
    _git_stdout,
    _head_reflog_exists,
    _looks_like_git_oid,
    _preflight_head_from_task,
    _preflight_head_present,
    _untracked_blob_exclude_reason,
    _workspace_patch_base,
    _write_patch_separator,
    build_workspace_patch,
    untracked_capture_veto_reason,
    write_workspace_patch_artifacts,
)

log = logging.getLogger(__name__)


HEADLESS_TASKS_DIR = pathlib.Path("state") / "headless_tasks"
ARTIFACTS_DIR = pathlib.Path("task_results") / "artifacts"
TASK_DRIVES_DIR = pathlib.Path("task_drives")


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
    # The budget-drive result is the sole durable authority for the root
    # post-task phase. A late child copy-back may enrich the result, but must not
    # replace a terminal canonical marker with the child's stale running mirror.
    existing_checkpoint = canonical_existing.get("root_phase_checkpoint")
    existing_post_task = (
        str(existing_checkpoint.get("post_task_synthesis") or "")
        if isinstance(existing_checkpoint, dict) else ""
    )
    if existing_post_task in {"completed", "degraded"}:
        child_checkpoint = payload.get("root_phase_checkpoint")
        merged_checkpoint = dict(child_checkpoint) if isinstance(child_checkpoint, dict) else {}
        # The child result owns the acceptance verdict.  The budget-drive copy
        # only owns the terminal post-task marker, which can settle before this
        # late copy-back.  Merging the whole parent checkpoint used to replace a
        # real PASS/DEGRADED verdict with the parent's provisional
        # ``not_required`` value.
        merged_checkpoint["post_task_synthesis"] = existing_post_task
        payload["root_phase_checkpoint"] = merged_checkpoint
        # F2: the same terminal checkpoint finalized the parent-owned accounting
        # (task_cost_finalized, exact subtree totals) on the canonical result; a
        # late copy-back of the child drive's stale root-only cost must not
        # overwrite it. total_rounds/prompt_tokens/completion_tokens ride the same
        # finalized record but are not in TASK_COST_META_FIELDS — named explicitly.
        for key in (*TASK_COST_META_FIELDS, "total_rounds", "prompt_tokens", "completion_tokens"):
            payload.pop(key, None)
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
        **payload,
    )


def _publish_child_verification_receipts(
    parent_drive_root: pathlib.Path, task_id: str, child_drive: pathlib.Path
) -> None:
    """Publish the child's durable ``verification_receipts.jsonl`` to the canonical root.

    ``verify_and_record`` appends receipts under the CHILD's isolated drive while
    the parent-side W2 readers (``control._get_task_result`` / ``wait_task``)
    resolve receipts against the canonical status root — without this copy the
    rows depend on a fail-silent artifact-sync side effect and rot entirely once
    the child drive is pruned. Whole-file atomic tmp+rename, not append-merge:
    receipts key on ``(drive_root, task_id)`` and this publish is the canonical
    file's only writer for a CHILD task_id, so the copy is collision-free; the
    append-only source makes re-publish (task_done + reaper/cancel re-checks) an
    idempotent refresh. Fail-soft: logged, never blocks finalization."""
    try:
        # Lazy import: ouroboros.outcomes imports from ouroboros.headless at module level.
        from ouroboros.outcomes import verification_receipts_path

        src = verification_receipts_path(child_drive, task_id, create=False)
        if not src.is_file():
            return
        dest = verification_receipts_path(parent_drive_root, task_id, create=True)
        if dest.exists() and os.path.samefile(src, dest):
            return  # shared-drive shape: already the canonical file
        tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
        shutil.copy2(src, tmp)
        replace_atomic(tmp, dest)
    except Exception:
        log.warning("Failed to publish child receipts for task %s", task_id, exc_info=True)


def _copy_child_artifacts_to_parent(
    parent_drive_root: pathlib.Path,
    task_id: str,
    child_drive: pathlib.Path,
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebase child-drive artifact files into the parent task artifact store."""

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
