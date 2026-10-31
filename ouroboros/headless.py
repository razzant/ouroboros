"""Isolated drives and patch artifacts for CLI/workspace runs; the gateway owns transport."""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import shutil
import subprocess  # noqa: F401
import tempfile  # noqa: F401
import threading  # noqa: F401
from hashlib import sha256
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

from ouroboros.contracts.task_constraint import normalize_task_constraint  # noqa: F401
from ouroboros.post_task_checkpoint import CANONICAL_CUSTODY_FIELDS, project_replica_task_result_fields
from ouroboros.workspace_file_outputs import _directory_direct_artifacts
from ouroboros.task_results import (
    cancellation_blocks_child_result, load_task_result, validate_task_id, write_task_result,
)
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.retention import task_result_timestamp as _timestamp_from_result
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
    pem_private_key_reason,
    untracked_capture_veto_reason,
    write_workspace_patch_artifacts,
)

log = logging.getLogger(__name__)


class _RefPublicationChanged(Exception):
    """Retry preparation outside the result lock against the new CURRENT refs."""


HEADLESS_TASKS_DIR = pathlib.Path("state") / "headless_tasks"
ARTIFACTS_DIR = pathlib.Path("task_results") / "artifacts"
TASK_DRIVES_DIR = pathlib.Path("task_drives")


# Pure patch rules keep their original identities on this facade; file I/O and
# untracked_capture_veto_reason remain owned by workspace_patch_capture.
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
    """Create a forked or empty execution drive; other modes keep the parent.

    Forks copy identity/world/registry and global knowledge except on Project
    tasks, which copy no knowledge file. Memory initializes empties.
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
    from ouroboros.task_custody import record_drive_arrival

    record_drive_arrival(parent, parent / HEADLESS_TASKS_DIR, task_id)  # the settlement lap looks at it alone
    return child


def _resolve_retention_days(retention_days: Optional[int]) -> int:
    """Only None reads the owner knob; explicit days reach age_cutoff unchanged.
    That shared owner floors at zero, which prunes everything before now."""
    from ouroboros.retention import get_gc_retention_days

    if retention_days is None:
        return get_gc_retention_days()
    return retention_days


def _effective_task_result(parent: pathlib.Path, task_id: str) -> Dict[str, Any]:
    """Prune reads the projected result; a projection failure reads canonical."""
    try:
        from ouroboros.task_status import load_effective_task_result

        return load_effective_task_result(parent, task_id) or {}
    except Exception:
        return load_task_result(parent, task_id) or {}


# How many drive settlements one off-loop pass attempts before yielding (each may copy and
# hash a whole child store); the caller carries the cursor so later passes continue.
DRIVE_SETTLEMENTS_PER_PASS = 16


def _prompt_settlement(result: Dict[str, Any]) -> bool:
    """A cancelled subagent's drive settles without waiting out retention: the cancel path
    used to delete it at once and now leaves that work to the off-loop pass."""
    return (str(result.get("status") or "").lower() == "cancelled"
            and str(result.get("delegation_role") or "") == "subagent")


def _drive_retention_rows(parent: pathlib.Path, names: List[str], stop: Any, *,
                          record: bool = True) -> Iterable[Tuple[str, Any]]:
    """Select only this layout's names in bounded transactions; never hold one across a yield.

    A selector/upgrade failure does not become absence: the existing strict reader
    handles each name, preserving missing/quarantined/unreadable-result outcomes.
    ``record=False`` reads without filling the retention projection (the preview).
    """
    from ouroboros import store
    from ouroboros.task_store import DRIVE_RETENTION_BATCH, drive_retention_facts

    for start in range(0, len(names), DRIVE_RETENTION_BATCH):
        batch = names[start:start + DRIVE_RETENTION_BATCH]
        try:
            facts = drive_retention_facts(parent, batch, stop=stop, record=record)
        except (store.StoreError, ValueError, TypeError):
            facts = {}
        for task_id in batch:
            yield task_id, facts.get(task_id)


def drive_candidates(base: pathlib.Path, parent: pathlib.Path, cutoff: float, stop: Any = None,
                     *, after: str = "", record: bool = True,
                     names: Optional[List[str]] = None) -> Iterable[Tuple[str, str, Optional[float], bool, Dict[str, Any]]]:
    """The selection half the prune and the free-space preview share: every drive folder under
    BASE in name order (wrapping after AFTER), each with the cheap canonical facts of its task row
    as ``(task_id, status, drive_retention_at, inside_wait, fact)``. ``inside_wait`` says the
    row is terminal but its retention instant is after CUTOFF; a missing row yields ``""`` as
    status and None as the instant (the strict reader decides later). A hint never authorizes
    removal: the prune re-reads the strict row before any settlement. ``record=False`` (the
    preview) classifies a legacy row in memory only; the prune fills the projection. ``names``
    (the drives whose own rows changed) replaces the listing: only those are selected."""
    names = _drive_names(base) if names is None else sorted(names)
    names = [name for name in names if name > after] + [name for name in names if name <= after]
    for task_id, fact in _drive_retention_rows(parent, names, stop, record=record):
        status, retention_at, inside = "", None, False
        usable = fact is not None and fact["body_present"] and fact["drive_retention_at"] is not None
        if usable:
            status = str(fact.get("status") or "").lower()
            raw = fact["drive_retention_at"]
            retention_at = float(raw) if raw else None
            if status in _FINAL_STATUSES and not _prompt_settlement(fact):
                try:
                    inside = (retention_at if retention_at is not None else (base / task_id).stat().st_mtime) > cutoff
                except OSError:
                    inside = False
        yield task_id, status, retention_at, inside, fact if usable else {}


def _drive_names(base: pathlib.Path) -> List[str]:
    """The drive folders of a layout in name order, from one listing (the entry's type, never a stat per name)."""
    with os.scandir(base) as entries:
        return sorted(entry.name for entry in entries if entry.is_dir())


def _earliest(known: Optional[float], instant: float) -> float:
    return float(instant) if known is None else min(float(known), float(instant))


def _moved_inputs(recorded: Any, basis: Dict[str, Any]) -> Optional[set]:
    """The inputs of the layout's recorded basis that differ from the current one, when each has an
    addressed reading: the terminal-write counter (rows were written terminal somewhere: look at the
    drives whose rows those are), the published arrivals (drive creators published: look at those
    drives), the layout directory together with the arrivals (the moved mtime is the published drives').
    ``None`` when another input moved, or the directory moved and nobody published: a whole walk."""
    if not isinstance(recorded, dict):
        return None
    moved = {key for key in set(recorded) | set(basis) if recorded.get(key) != basis.get(key)}
    explained = (moved <= {"terminal", "arrivals", "directory"} and ("directory" not in moved or "arrivals" in moved)
                 and ("terminal" not in moved or isinstance(recorded.get("terminal"), int)))
    return moved if explained else None


def _changed_drives(base: pathlib.Path, parent: pathlib.Path, since: int) -> List[str]:
    """The drives of BASE whose own task rows were written terminal since the counter stood at SINCE: one
    indexed read of those rows (``task_store.terminal_writes_since``) and one stat per row, never a listing."""
    from ouroboros.task_store import terminal_writes_since

    changed = []
    for task_id in terminal_writes_since(parent, since):
        try:
            validate_task_id(task_id)
        except ValueError:
            continue
        if (base / task_id).is_dir():
            changed.append(task_id)
    return changed


def _prune_drives(base: pathlib.Path, parent: pathlib.Path, *, drive_of: Any, not_terminal: str,
                  retention_days: Optional[int], now: Optional[float], live: Any, guard: Any, stop: Any,
                  budget: Optional[int], after: str, extra_checks: Any, rested: Any = None,
                  only: Optional[Any] = None, progress: Any = None, rest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The prune both layouts share: cheap per-name facts select possible work;
    fresh strict canonical status/date and existing checks still decide eligibility.
    Settlement through ``task_custody.settle_child_drive`` runs in name order after ``after`` (wrapping), at
    most ``budget`` attempts per call; ``cursor`` is the last attempted drive, ``deferred`` the unreached rest
    (no verdict is read and no row decoded for it). ``rested`` is the caller's durable verdicts
    (``task_custody.rested_verdict``), updated in place, or a callable that loads them: a pass that walks reads
    them, one that looks at named drives only when one of them reaches a verdict (a running or young drive
    never does), one at rest never. ``rest`` is the layout's durable rest, updated in place: the inputs
    (``task_custody.layout_basis``) and the earliest retention instant of a drive it left young; while both
    hold, the pass lists nothing (``rested`` in the report). When only inputs with an addressed reading moved
    (``_moved_inputs``: the terminal-write counter, rows written terminal somewhere; the published arrivals,
    with or without the directory's mtime they moved), the pass looks only at the drives whose own rows
    those were (``_changed_drives``) and at the drives published (``task_custody.arrived_drives``; the
    report's ``arrivals``, which the caller forgets in the transaction recording the pass): a completion
    elsewhere or a new drive leaves the rest standing under the moved inputs, still without a listing. A
    whole pass that attempted nothing records the rest; any other pass clears it.
    ``only`` restricts the pass to those task ids (the free action's previewed folders; every other drive is
    untouched, no rest is kept); ``progress`` is called with the report after each settlement attempt."""
    from ouroboros.retention import age_cutoff
    from ouroboros.task_custody import (arrived_drives, forget_departed, layout_basis, record_rest, rested_verdict,
                                        settle_child_drive)

    days = _resolve_retention_days(retention_days)
    cutoff = age_cutoff(days, now)
    report: Dict[str, Any] = {"retention_days": days, "scanned": 0, "pruned": [], "skipped": [], "errors": [],
                              "deferred": [], "cursor": after, "rested": False, "arrivals": []}
    if not base.is_dir():
        return report
    chosen = None if only is None else {str(task_id) for task_id in only}
    basis = layout_basis(parent, base, days) if rest is not None and chosen is None else None
    young_due = basis is not None and rest.get("youngest") is not None and float(rest["youngest"]) <= cutoff
    if basis is not None and rest.get("basis") == basis and not young_due:
        report["rested"] = True
        return report  # nothing listed, decoded, copied or hashed: what the last whole pass saw still stands
    names = None
    if basis is not None:
        report["arrivals"] = arrived_drives(parent, base)  # looked at by this pass, alone or inside the walk
    if basis is not None and not young_due and (moved := _moved_inputs(rest.get("basis"), basis)) is not None:
        changed = _changed_drives(base, parent, rest["basis"]["terminal"]) if "terminal" in moved else []
        names = sorted({*changed, *report["arrivals"]})
        if not names:
            rest["basis"] = basis  # the rows written and the drives published were none of this layout's: its rest stands
            report["rested"] = True
            return report
    def verdicts() -> Any:  # the durable verdicts, decoded by the first drive whose verdict is compared or kept
        nonlocal rested
        rested = rested() if callable(rested) else rested
        return rested

    if names is None:
        forget_departed(verdicts(), base.name, _drive_names(base))
    attempts, youngest = 0, (None if names is None else rest.get("youngest"))
    for task_id, status, retention_at, inside_wait, fact in drive_candidates(base, parent, cutoff, stop, after=after,
                                                                               names=names):
        if chosen is not None and task_id not in chosen:
            continue
        task_dir = base / task_id
        report["scanned"] += 1
        try:
            validate_task_id(task_id)
            if fact:
                if status not in _FINAL_STATUSES:
                    report["skipped"].append({"task_id": task_id, "reason": not_terminal, "status": status})
                    continue
                if inside_wait:
                    youngest = _earliest(youngest, retention_at if retention_at is not None else task_dir.stat().st_mtime)
                    report["skipped"].append({"task_id": task_id, "reason": "younger_than_retention"})
                    continue
            if (stop is not None and stop()) or (budget is not None and attempts >= budget):
                report["deferred"].append(task_id)  # unreached this pass: its turn comes after the cursor
                continue
            if verdict := rested_verdict(verdicts(), parent, task_id, task_dir, drive_of(task_dir)):
                report["skipped"].append({"task_id": task_id, "reason": "custody_unchanged", "retained": verdict.get("reason")})
                continue
            # A hint never authorizes removal. In particular a date changed after
            # selection must be checked here; settlement rechecks custody, not age.
            result = load_task_result(parent, task_id, strict=True) or {}
            status = str(result.get("status") or "").lower()
            if status not in _FINAL_STATUSES:
                report["skipped"].append({"task_id": task_id, "reason": not_terminal, "status": status})
                continue
            instant = _timestamp_from_result(result, task_dir.stat().st_mtime)
            if not _prompt_settlement(result) and instant > cutoff:
                youngest = _earliest(youngest, instant)
                report["skipped"].append({"task_id": task_id, "reason": "younger_than_retention"})
                continue
            skip = extra_checks(task_dir, result)
            if skip is not None:
                report["skipped"].append({"task_id": task_id, **skip})
                continue
            attempts += 1
            report["cursor"] = task_id
            if (outcome := settle_child_drive(parent, task_id, drive_of(task_dir), live=live, guard=guard, stop=stop,
                                              report=report))["status"] == "removed":
                report["pruned"].append({"task_id": task_id, "path": str(task_dir)})
            record_rest(verdicts(), base.name, parent, task_id, task_dir, drive_of(task_dir), outcome)
            if progress is not None:
                progress(report)
        except Exception as exc:
            report["errors"].append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    if basis is not None:
        rest.clear()
        if not (attempts or report["deferred"] or report["errors"] or (stop is not None and stop())):
            rest.update({"basis": basis, "youngest": youngest})  # a whole pass with nothing to do: its inputs stand
    return report


def prune_headless_task_drives(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
    live: Any = None,
    guard: Any = None,
    stop: Any = None,
    budget: Optional[int] = None,
    after: str = "",
    rested: Any = None,
    only: Optional[Any] = None,
    progress: Any = None,
    rest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prune terminal child drives past retention (a cancelled subagent's at once). Removal is
    ``task_custody.settle_child_drive``'s decision alone: ``live`` is the supervisor's probe (without it nothing
    is removed), ``guard`` its interlock, ``stop`` the generation's close, ``budget``/``after`` the pass bound/cursor,
    ``rested`` the durable verdicts of retained drives, ``only``/``progress`` the free action's selection and
    progress callback (``_prune_drives``)."""
    parent = pathlib.Path(parent_drive_root)

    def checks(task_dir: pathlib.Path, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        artifact_status = str(result.get("artifact_status") or "").lower()
        if artifact_status and artifact_status not in ARTIFACT_TERMINAL_STATUSES:
            return {"reason": "artifacts_not_terminal", "artifact_status": artifact_status}
        known_child = str(result.get("child_drive_root") or result.get("headless_child_drive_root")
                          or result.get("drive_root") or "").strip()
        if known_child and pathlib.Path(known_child).resolve(strict=False) != (task_dir / "data").resolve(strict=False):
            return {"reason": "child_drive_mismatch"}
        return None

    return _prune_drives(parent / HEADLESS_TASKS_DIR, parent, drive_of=lambda task_dir: task_dir / "data",
                         not_terminal="parent_not_terminal", retention_days=retention_days, now=now, live=live,
                         guard=guard, stop=stop, budget=budget, after=after, extra_checks=checks, rested=rested,
                         only=only, progress=progress, rest=rest)


def prune_task_drives(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
    live: Any = None,
    guard: Any = None,
    stop: Any = None,
    budget: Optional[int] = None,
    after: str = "",
    rested: Any = None,
    only: Optional[Any] = None,
    progress: Any = None,
    rest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prune direct-task scratch drives past retention (``settle_child_drive`` decides;
    same knobs as ``prune_headless_task_drives``)."""
    parent = pathlib.Path(parent_drive_root)
    return _prune_drives(parent / TASK_DRIVES_DIR, parent, drive_of=lambda task_dir: task_dir,
                         not_terminal="task_not_terminal", retention_days=retention_days, now=now, live=live,
                         guard=guard, stop=stop, budget=budget, after=after, extra_checks=lambda *_a: None,
                         rested=rested, only=only, progress=progress, rest=rest)


def prune_task_trees(
    parent_drive_root: pathlib.Path,
    *,
    retention_days: Optional[int] = None,
    now: Optional[float] = None,
    exclude_root_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Prune ephemeral task-tree ledgers once the root is terminal or absent and
    older than GC retention; durable project memory is outside this plane."""

    from ouroboros.retention import age_cutoff, keep_until

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
        if root_id in (exclude_root_ids or ()):
            report["skipped"].append({"root_task_id": root_id, "reason": "source_recovery_pending"})
            continue
        try:
            dir_mtime = tree_dir.stat().st_mtime
            result = _effective_task_result(parent, root_id)
            status = str(result.get("status") or "").lower()
            if status and status not in _FINAL_STATUSES:
                report["skipped"].append({"root_task_id": root_id, "reason": "root_not_terminal", "status": status})
                continue
            if _timestamp_from_result(result, dir_mtime) > cutoff:
                keep_until(report, _timestamp_from_result(result, dir_mtime), days)
                report["skipped"].append({"root_task_id": root_id, "reason": "younger_than_retention"})
                continue
            from ouroboros.owner_keep import gate

            with gate(parent, root_id, timeout_sec=0) as why:
                if why:
                    report["skipped"].append({"root_task_id": root_id, "reason": why})
                    continue
                shutil.rmtree(tree_dir)
            report["pruned"].append({"root_task_id": root_id, "path": str(tree_dir)})
        except Exception as exc:
            report["errors"].append({"root_task_id": root_id, "error": f"{type(exc).__name__}: {exc}"})
    return report


def remove_subagent_task_drive(parent_drive_root: pathlib.Path, task_id: str, *, live: Any = None,
                               guard: Any = None, admission_rollback: bool = False) -> bool:
    """Remove TASK's own drives through ``task_custody.settle_child_drive`` (the one deletion owner):
    ``live`` is the supervisor's probe, ``guard`` its interlock, ``admission_rollback`` frees a never-started drive."""
    from ouroboros.task_custody import own_child_drives, settle_child_drive

    try:
        drives = own_child_drives(parent_drive_root, validate_task_id(task_id))
    except Exception:
        return False
    return any([settle_child_drive(parent_drive_root, task_id, drive, live=live, guard=guard,
                                   admission_rollback=admission_rollback)["status"] == "removed"
                for drive in drives])


# How long a publisher waits for the task's custody lock before answering ``CustodyBusy``.
PUBLICATION_LOCK_SEC = 30.0


def copy_child_task_result(parent_drive_root: pathlib.Path, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy a child-drive task result back to the parent data root under the task's custody lock
    (``task_custody.task_custody_lock``) from the child read to the row write, so a settlement never moves
    the drive under a copy in flight; a settled (gone) drive publishes nothing; a busy lock is ``CustodyBusy``."""
    from ouroboros.observability import child_ref_promotion_scope
    from ouroboros.task_custody import CustodyBusy, task_custody_lock

    with child_ref_promotion_scope():
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
        with task_custody_lock(parent_drive_root, task_id, timeout_sec=PUBLICATION_LOCK_SEC) as locked:
            if not locked:
                raise CustodyBusy(f"custody lock of {task_id} is held by another publisher")
            canonical_existing = load_task_result(parent_drive_root, task_id) or {}
            if cancellation_blocks_child_result(canonical_existing):
                return canonical_existing  # cancelled while this publisher waited: the child's authority is declined
            return _copy_child_task_result_locked(parent_drive_root, task, task_id, canonical_existing, child_drive)


def _copy_child_task_result_locked(parent_drive_root: pathlib.Path, task: Dict[str, Any], task_id: str,
                                   canonical_existing: Dict[str, Any], child_drive: pathlib.Path) -> Optional[Dict[str, Any]]:
    child_result = load_task_result(child_drive, task_id)
    if not isinstance(child_result, dict):
        return None
    # Adopt the answer and deliverables now. Historical sources keep their exact
    # references and their child drive until the existing off-loop custody pass
    # verifies the archive. History size must not delay task_done or startup.
    from ouroboros.history_retention import prepare_result_retention

    child_result, ref_promotion = prepare_result_retention(
        pathlib.Path(parent_drive_root), child_drive, task_id, child_result)
    _publish_child_verification_receipts(parent_drive_root, task_id, child_drive)
    child_status = str(child_result.pop("status", None) or "completed")
    child_result.pop("task_id", None)
    child_result["child_ref_promotion"] = ref_promotion
    if isinstance(child_result.get("artifacts"), list):
        try:
            from ouroboros.outcomes import artifact_bundle_from_result
            child_result["artifact_bundle"] = artifact_bundle_from_result(child_result)
        except Exception:
            child_result.pop("artifact_bundle", None)
    child_result.setdefault("headless_child_drive_root", str(child_drive))
    if child_status in _FINAL_STATUSES and task_requires_workspace_capture(task):
        artifact_status = str(canonical_existing.get("artifact_status") or "").strip().lower()
        if artifact_status in ARTIFACT_TERMINAL_STATUSES | {ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING}:
            child_result["artifacts"] = _merge_artifacts(
                list(canonical_existing.get("artifacts") or []), list(child_result.get("artifacts") or []))
            child_result.update({key: canonical_existing[key] for key in _ARTIFACT_LIFECYCLE_FIELDS
                            if key in canonical_existing})
        else:
            child_result["artifact_status"] = ARTIFACT_STATUS_FINALIZING
        child_result["child_status"] = child_status

    return _retry_child_task_refs_locked(parent_drive_root, child_drive, task_id,
                                         replica={**child_result, "task_id": task_id, "status": child_status})


class _GenerationClosed(Exception):
    """The maintenance generation closed before this publication's commit."""


def retry_child_task_refs(parent: pathlib.Path, child: pathlib.Path, task_id: str,
                          *, replica: Optional[Dict[str, Any]] = None, stop: Any = None) -> Dict[str, Any]:
    """One optimistic publisher for pending CURRENT refs and prepared copyback, under the task's custody
    lock (``CustodyBusy`` when held). ``stop()`` fences every file it promotes (``publication_fence``) and is
    re-asked at the commit: a closed generation starts no further write and returns CURRENT."""
    from ouroboros.task_custody import CustodyBusy, publication_fence, task_custody_lock

    with task_custody_lock(parent, task_id, timeout_sec=PUBLICATION_LOCK_SEC) as locked, publication_fence(stop):
        if not locked:
            raise CustodyBusy(f"custody lock of {task_id} is held by another publisher")
        return _retry_child_task_refs_locked(parent, child, task_id, replica=replica, stop=stop)


def _retry_child_task_refs_locked(parent: pathlib.Path, child: pathlib.Path, task_id: str,
                                  *, replica: Optional[Dict[str, Any]] = None, stop: Any = None) -> Dict[str, Any]:
    """Normal copyback supplies its prepared replica without history I/O; retry has
    no replica and never reads an old child body; a changed CURRENT basis repeats preparation."""
    from ouroboros.observability import (
        _has_pending_ref_promotion,
        child_ref_promotion_scope, promote_child_task_ref_patch,
    )
    with child_ref_promotion_scope():
        while True:
            source = load_task_result(parent, task_id, strict=True) or {}
            if stop is not None and stop():
                return source  # a closed generation starts no file promotion, not only no commit
            if replica is None and not source:
                raise ValueError("pending child-ref authority is missing")
            if replica is None:
                if not _has_pending_ref_promotion(source.get("child_ref_promotion")):
                    return source
                patch = promote_child_task_ref_patch(parent, child, task_id, source)
                basis = {key: source.get(key) for key in patch}
            else:
                source = {**source, **project_replica_task_result_fields(source, replica)}
                review = source.get("review_projection")
                promotion = copy.deepcopy(replica["child_ref_promotion"])
                patch = {"child_ref_promotion": promotion}
                if isinstance(review, dict):
                    patch["review_projection"] = review
                basis = {"review_projection": review}

            def project(current: dict, _incoming: dict) -> Optional[dict]:
                if stop is not None and stop():
                    raise _GenerationClosed()  # re-asked at the commit, not only before the walk
                if replica is not None and cancellation_blocks_child_result(current):
                    return None  # cancelled under the row lock: the child enriches nothing, not even by replica
                selected = {**current, **project_replica_task_result_fields(current, replica)} if replica is not None else current
                if {key: selected.get(key) for key in basis} != basis:
                    raise _RefPublicationChanged()
                if replica is None and all(key in current and current[key] == value for key, value in patch.items()):
                    return None  # an unchanged retry rewrites nothing: no fresh updated_at/mtime (#1305)
                if replica is not None and current.get("status") in _FINAL_STATUSES \
                        and selected.get("status") != current["status"]:
                    # A settled canonical row keeps its outcome; the child's is child_status.
                    selected = {**selected, **{key: current[key] for key in ("status", "result", "error", "ts")
                                               if key in current}, "child_status": replica.get("status")}
                return {**(selected if replica is not None else {}), **patch, "status": selected["status"]}

            try:
                return write_task_result(parent, task_id, source["status"],
                                         _field_projector=project, strict_existing_dict=True,
                                         _terminal_time_source=replica,
                                         **{key: value for key, value in (replica or {}).items()
                                            if key not in {"status", "task_id", *CANONICAL_CUSTODY_FIELDS}})
            except _RefPublicationChanged:
                continue
            except _GenerationClosed:
                return source


def _child_result_adopted(child: pathlib.Path, result: Dict[str, Any]) -> bool:
    promotion = result.get("child_ref_promotion")
    source = str(result.get("headless_child_drive_root") or "")
    return bool(
        "result" in result and isinstance(promotion, dict)
        and type(promotion.get("schema_version")) is int and promotion["schema_version"] == 1
        and promotion.get("status") in {"complete", "incomplete"}
        and source and pathlib.Path(source).resolve(strict=False) == child.resolve(strict=False)
    )


def terminal_task_files_ready(canonical_root: pathlib.Path, task: Dict[str, Any], result: Any) -> bool:
    """Read-only CURRENT readiness, not proof that every ref is available.
    Split tasks require adopted body facts, not an early terminal checkpoint.
    """
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not isinstance(result, dict) or not task_id or str(result.get("task_id") or "") != task_id:
        return False
    if str(result.get("status") or "") not in _FINAL_STATUSES:
        return False
    if cancellation_blocks_child_result(result):
        return True
    try:
        child = _child_drive_from_task(task)
        if child is not None and child.resolve(strict=False) != pathlib.Path(canonical_root).resolve(strict=False):
            if not _child_result_adopted(child, result):
                return False
    except (OSError, ValueError, RuntimeError):
        return False
    return not task_requires_workspace_capture(task) or (
        str(result.get("artifact_status") or "").strip().lower() in ARTIFACT_TERMINAL_STATUSES
    )


def prepare_terminal_task_files(canonical_root: pathlib.Path, task: Dict[str, Any]) -> Dict[str, Any]:
    """Finish one file-save attempt without turning an I/O failure into a worker crash.

    The dispatcher re-reads CURRENT; pending refs keep custody. terminal_source_present is True after a strict terminal read,
    False for absence/nonterminal, None for unknown; later write failure preserves that observation, never inventing a source.
    """
    task_id = str(task.get("id") or task.get("task_id") or "")
    report: Dict[str, Any] = {"task_id": task_id, "result": None, "error": "", "terminal_source_present": None}
    from ouroboros.task_custody import CustodyBusy
    from ouroboros.install_identity import StoreForeign, require_owned_root

    try:
        root = require_owned_root(canonical_root)
        child = _child_drive_from_task(task)
        if child is not None:
            require_owned_root(child)
        current = load_task_result(root, task_id, strict=True) or {}
        cancelled = cancellation_blocks_child_result(current)
        adopted = not cancelled and child is not None and _child_result_adopted(child, current)
        source = current
        if child is not None and not adopted and not cancelled:
            source = load_task_result(child, task_id, strict=True)
        report["terminal_source_present"] = bool(source and str(source.get("status") or "") in _FINAL_STATUSES)
        if not report["terminal_source_present"]:
            raise ValueError("terminal task source is missing or not settled")
        if not adopted and child is not None and not cancelled:
            current = copy_child_task_result(root, {**task, "id": task_id}) or current
        if str(current.get("status") or "") not in _FINAL_STATUSES:
            raise ValueError("terminal task result is missing or not settled")
        # Workspace capture and a workspace-free child's memory export share the
        # existing finalizer. Inline turns have no workspace capture obligation.
        memory_export = child is not None and _workspace_root_from_task(task) is None and not task_is_readonly_subagent(task)
        if (not cancelled and (task_requires_workspace_capture(task) or memory_export)
                and str(current.get("artifact_status") or "").strip().lower() not in ARTIFACT_TERMINAL_STATUSES):
            finalize_task_artifacts(root, {**task, "id": task_id})
        report["result"] = load_task_result(root, task_id, strict=True)
        from ouroboros.obligations import drive_finished
        drive_finished(root, task, report["result"])
        # Derived inventories need not outlive terminal file adoption.
        if child is not None and child.resolve(strict=False) != root.resolve(strict=False):
            shutil.rmtree(child / "state" / "code_intel", ignore_errors=True)
    except StoreForeign as exc:
        report.update(error=str(exc), reason_code=exc.reason, reason_detail=str(exc))
        log.warning("Terminal file preparation for %s refused: %s", task_id, exc)
    except CustodyBusy as exc:
        # Another publisher (a settlement or copy-back) holds the store: the next attempt
        # completes the same work; nothing failed and no failure is stamped.
        report["error"] = f"CustodyBusy: {exc}"
        log.warning("Terminal file preparation for %s waits: %s", task_id, exc)
    except Exception as exc:
        from ouroboros.observability import redact_projection
        report["error"] = str(redact_projection(f"{type(exc).__name__}: {exc}").value)
        log.warning("Terminal file preparation failed for %s: %s", task_id, report["error"])
        # Only a real terminal source may stamp an artifact failure; absent/nonterminal source follows crash recovery.
        if report["terminal_source_present"] is not True:
            return report
        try:
            existing = load_task_result(canonical_root, task_id, strict=True)
            if existing:
                from ouroboros.outcomes import artifact_bundle_from_result
                fields = dict(artifact_status=ARTIFACT_STATUS_FAILED, artifact_error=report["error"],
                              artifact_finalized_at=utc_now_iso())
                provisional = {**existing, **fields}
                provisional.pop("artifact_bundle", None)
                fields["artifact_bundle"] = artifact_bundle_from_result(provisional)
                report["result"] = write_task_result(
                    canonical_root, task_id, existing["status"],
                    _field_projector=lambda live, fields: {**fields, "status": live["status"]},
                    strict_existing_dict=True, **fields,
                )
        except Exception:
            log.warning("Terminal file failure could not be persisted for %s", task_id, exc_info=True)
    return report


def _publish_child_verification_receipts(
    parent_drive_root: pathlib.Path, task_id: str, child_drive: pathlib.Path
) -> None:
    """Union child verification receipts with canonical delegation_zero_run rows.
    Exact-row dedup makes re-entry idempotent; failures log without blocking."""
    try:
        from ouroboros.outcome_receipt_store import publish_verification_receipt_union

        publish_verification_receipt_union(
            parent_drive_root, task_id, child_drive,
        )
    except Exception:
        log.warning("Failed to publish child receipts for task %s", task_id, exc_info=True)


def _copy_child_artifacts_to_parent(
    parent_drive_root: pathlib.Path, task_id: str,
    child_drive: pathlib.Path, artifacts: List[Dict[str, Any]],
    *, promotion: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Copy back at child-store relpaths (nested rows gain ``relpath``). Immutable
    bytes stay exact; differing mutable copies are versioned first. A failure
    keeps the original row and a pending reference."""
    from ouroboros.artifacts import (
        ArtifactIdentityError, _archive_previous_artifact_version, copy_artifact_file, stream_artifact_file,
    )
    from ouroboros.outcome_receipt_store import is_verification_receipts_path

    parent_dir = task_artifacts_dir(parent_drive_root, task_id)
    parent_base = parent_dir.resolve(strict=False)
    child_base = task_artifacts_dir(child_drive, task_id, create=False).resolve(strict=False)
    rebased: List[Dict[str, Any]] = []
    for artifact in artifacts:
        item = dict(artifact)
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            rebased.append(item)
            continue
        src = pathlib.Path(raw_path)
        source = src = (src if src.is_absolute() else child_drive / raw_path).resolve(strict=False)
        if is_verification_receipts_path(child_drive, task_id, src):
            # Receipt union has its own locked writer; never replace its rows.
            continue
        expected = item if item.get("immutable") else None
        dest = (src if src.is_relative_to(parent_base) else
                parent_dir / (src.relative_to(child_base) if src.is_relative_to(child_base) else src.name))
        canonical_candidate = dest
        if dest != src and expected is not None and dest.exists() and dest.resolve(strict=False) != src:
            try:
                stream_artifact_file(dest, expected=item)
                src = dest  # Exact canonical bytes already survive this copy-back.
            except OSError:
                dest = dest.with_name(f"{src.stem}_{sha256(str(src).encode('utf-8')).hexdigest()[:8]}{src.suffix}")
        try:
            if expected is None and dest != src and dest.is_file() and not dest.is_symlink():
                _archive_previous_artifact_version(pathlib.Path(parent_drive_root), task_id, dest, src)
            measured = copy_artifact_file(src, dest, expected=expected)
        except OSError as exc:
            item.update(copy_status="failed", copy_error=f"{type(exc).__name__}: {exc}")
            if promotion is not None:
                pending = {"path": str(src), "kind": "task_artifact", "reason": item["copy_error"]}
                if expected is not None and isinstance(exc, ArtifactIdentityError):
                    pending.update(failure_kind="immutable_identity_mismatch", source_path=str(source),
                                   destination_path=str(dest), canonical_path=str(canonical_candidate),
                                   sha256=item.get("sha256"), size=item.get("size"),
                                   failed_path=str(exc.source_path), failed_stamp=list(exc.source_stamp))
                promotion["pending_refs"].append(pending)
            rebased.append(item)
            continue
        item.pop("copy_status", None)
        item.pop("copy_error", None)
        item.update(path=str(dest), name=str(item.get("name") or dest.name), **measured)
        relpath = dest.resolve(strict=False).relative_to(parent_base).as_posix() \
            if dest.resolve(strict=False).is_relative_to(parent_base) else ""
        item.update({"relpath": relpath} if "/" in relpath else {})
        rebased.append(item)
    return rebased


def task_is_readonly_subagent(task: Dict[str, Any]) -> bool:
    """The shared completion/reaper exception: readonly children need no artifacts."""
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


def task_requires_workspace_capture(task: Dict[str, Any]) -> bool:
    """Capture applicability shared by completion, preparation and status readers.

    The actual execution owner wins over historical direct/Presence origin.
    Source verification, cancellation and split-result adoption precede this test.
    """
    from ouroboros.task_carrier import inline_execution

    return (_workspace_root_from_task(task) is not None
            and not task_is_readonly_subagent(task) and not inline_execution(task))


def _build_deliverable_manifest(
    workspace_root: pathlib.Path, task_id: str, project_id: str
) -> Dict[str, Any]:
    """List genesis outputs with streamed hashes, junk excluded, symlinks unfollowed.
    Read/change gaps are disclosed, not delivery failures; actual copies stay strict.
    """
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.store import is_wal_index

    contents: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []  # a WAL index: shared memory never read, rebuilt from the log
    def failed(exc: OSError) -> None:
        path = pathlib.Path(exc.filename) if exc.filename else workspace_root
        contents.append({"rel": path.relative_to(workspace_root).as_posix(), "type": "directory",
                         "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
    for root, dirs, files in os.walk(workspace_root, onerror=failed):
        dirs[:] = sorted(d for d in dirs if d not in _TOP_LEVEL_EXCLUDE_DIRS and d != ".git")
        for fname in sorted(files):
            fpath = pathlib.Path(root) / fname
            entry = {"rel": fpath.relative_to(workspace_root).as_posix()}
            if is_wal_index(fpath):
                excluded.append({**entry, "reason": "SQLite WAL index"})
                continue
            try:
                if fpath.is_symlink():
                    entry.update(symlink=True, sha256="")
                else:
                    entry.update(stream_artifact_file(fpath))
            except OSError as exc:
                entry.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            contents.append(entry)
    gap_count = sum(item.get("status") == "unavailable" for item in contents)
    return {
        "schema_version": 1, "task_id": task_id, "project_id": project_id,
        "project_root": str(workspace_root), "created_at": utc_now_iso(),
        "file_count": sum(item.get("type") != "directory" for item in contents),
        "complete": gap_count == 0, "gap_count": gap_count,
        "truncated": False, "contents": contents, **({"excluded": excluded} if excluded else {}),
    }


def _file_artifact(kind: str, path: pathlib.Path, **facts: Any) -> Dict[str, Any]:
    """One row shape for an artifact written straight into the task artifact dir, measured by the one
    owner as published: retaining the same bytes changes no identity (#1969); producer facts stay."""
    from ouroboros.artifacts import stream_artifact_file
    return {"kind": kind, "name": path.name, "path": str(path), **stream_artifact_file(path), **facts}


def finalize_task_artifacts(parent_drive_root: pathlib.Path, task: Dict[str, Any],
                            *, lock_timeout_sec: Optional[float] = None) -> List[Dict[str, Any]]:
    """Publish under the custody lock shared by copy-back, ref retry, settlement and mailbox cleanup.
    Publishers never place/list one task's files concurrently; an exhausted lock budget is ``CustodyBusy``."""
    from ouroboros.task_custody import CustodyBusy, task_custody_lock

    task_id = str(task.get("id") or "")
    if not task_id:
        return []
    budget = PUBLICATION_LOCK_SEC if lock_timeout_sec is None else lock_timeout_sec
    with task_custody_lock(parent_drive_root, task_id, timeout_sec=budget) as locked:
        if not locked:
            raise CustodyBusy(f"custody lock of {task_id} is held by another publisher")
        return _finalize_task_artifacts_locked(parent_drive_root, task, task_id)


def _finalize_task_artifacts_locked(parent_drive_root: pathlib.Path, task: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
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
            direct = _directory_direct_artifacts(workspace_root, artifact_dir, task, existing)
            patch_artifacts, manifest = direct if direct is not None else write_workspace_patch_artifacts(
                workspace_root, artifact_dir, task=task,
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
            try:
                row = _file_artifact("workspace_patch_manifest", manifest_path, workspace_root=str(workspace_root))
            except OSError as unmeasured:  # a read-back fault never replaces the capture failure it records
                note = f"failure manifest unmeasured: {type(unmeasured).__name__}: {unmeasured}"
                artifact_error = f"{artifact_error}; {note}"
                row = {"kind": "workspace_patch_manifest", "name": manifest_path.name, "path": str(manifest_path),
                       "workspace_root": str(workspace_root), "measured": False, "errors": [note]}
            artifacts.append(row)

    child_drive = _child_drive_from_task(task)
    if child_drive is not None:
        try:
            export_path = artifact_dir / "memory_export.json"
            atomic_write_json(export_path, build_memory_export(child_drive, task), trailing_newline=True)
            artifacts.append(_file_artifact("memory_export", export_path,
                                            memory_mode=str(task.get("memory_mode") or "")))
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
            artifacts.append(_file_artifact(
                "deliverable_manifest", manifest_path,
                file_count=int(dm.get("file_count") or 0), truncated=bool(dm.get("truncated")),
                complete=dm["complete"], gap_count=dm["gap_count"],
                errors=([f"Automatic workspace listing is partial: {dm['gap_count']} read gaps."]
                        if dm["gap_count"] else []),
                workspace_root=str(workspace_root),
            ))
        except Exception as exc:
            artifact_status = ARTIFACT_STATUS_FAILED
            artifact_error = f"Deliverable manifest failed: {type(exc).__name__}: {exc}"
            log.warning("deliverable_manifest build failed for %s", task_id, exc_info=True)

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
                        stub = fields.get("verification_ledger")
                        if isinstance(stub, dict) and stub.get("omitted_to_artifact") and isinstance(refreshed_artifact_ledger.get("summary"), dict):
                            fields["verification_ledger"] = {**stub, "summary": dict(refreshed_artifact_ledger["summary"])}
                except Exception:
                    log.debug("Failed to refresh verification ledger artifact for task %s", task_id, exc_info=True)
            # The loop rewrote the ledger file in place, so the bundle computed
            # before it no longer describes the bytes on disk.
            fields["artifact_bundle"] = artifact_bundle_from_result(provisional)
        except Exception:
            log.warning("Artifact bundle/ledger refresh failed for task %s", task_id, exc_info=True)
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
    project = bool(str(project_id or "").strip())
    # A project child inherits none of the global knowledge topics or index.
    paths = ["identity.md", "WORLD.md", "registry.md"]
    for rel in paths:
        src = parent_memory / rel
        if src.is_file():
            dst = child_memory / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    src_knowledge = parent_memory / "knowledge"
    dst_knowledge = child_memory / "knowledge"
    if not project and src_knowledge.is_dir():
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

    drop = drop_kinds or set()
    key_for = lambda item: (
        str(item.get("kind") or ""),
        str(item.get("name") or pathlib.Path(str(item.get("path") or "")).name),
    )
    keys = {key_for(item) for item in new_items if isinstance(item, dict)}
    return [item for item in existing if isinstance(item, dict)
            and key_for(item)[0] not in drop and key_for(item) not in keys] + new_items


__all__ = [
    "ARTIFACT_STATUS_FAILED",
    "ARTIFACT_STATUS_FINALIZING",
    "ARTIFACT_STATUS_PENDING",
    "ARTIFACT_STATUS_READY",
    "build_memory_export",
    "build_workspace_patch",
    "copy_child_task_result",
    "prepare_terminal_task_files",
    "terminal_task_files_ready",
    "finalize_task_artifacts",
    "task_is_readonly_subagent",
    "task_requires_workspace_capture",
    "prepare_task_drive",
    "drive_candidates",
    "prune_headless_task_drives",
    "prune_task_drives",
    "task_artifacts_dir",
    "task_state_dir",
    "write_workspace_patch_artifacts",
    "write_workspace_preflight_artifact",
]
