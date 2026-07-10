"""``integrate_subagent_patch``: the parent's manifest-first integration tool.

A mutative (acting) subagent returns its changes as a ``workspace.patch`` artifact
(produced by headless finalization, a git diff against the child's base commit).
The parent decides what to do with it — accept one (best-of-N), synthesize several,
or reject — and this tool APPLIES the chosen patch into the parent's active repo or
worktree. The parent stays the sole committer: this stages changes but never
commits; the parent reviews and runs ``commit_reviewed`` itself.

Routing is top-only: ``target_root`` defaults to ``ctx.active_repo_dir()`` — the
live repo for the root agent, or the parent's own worktree for a nested acting
parent, so descendants bubble their patches up one level at a time.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from typing import Any, Dict, List, Tuple, Union

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.task_results import load_task_result
from ouroboros.review_state import invalidate_advisory_after_mutation
from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.config import get_runtime_mode
from ouroboros.headless import ARTIFACT_STATUS_READY_WITH_CHANGES
from ouroboros.utils import atomic_write_json, utc_now_iso


def _candidate_drive_roots(ctx: ToolContext) -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    seen = set()
    meta = getattr(ctx, "task_metadata", {})
    meta_budget = meta.get("budget_drive_root") if isinstance(meta, dict) else ""
    for raw in (
        getattr(ctx, "drive_root", None),
        getattr(ctx, "budget_drive_root", None),
        meta_budget,
    ):
        if not raw:
            continue
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        roots.append(pathlib.Path(raw))
    return roots


def _locate_child_patch(
    ctx: ToolContext, child_task_id: str
) -> Union[str, Tuple[pathlib.Path, Dict[str, Any], Dict[str, Any]]]:
    roots = _candidate_drive_roots(ctx)
    for root in roots:
        try:
            art_dir = task_artifact_dir_path(root, child_task_id)
        except Exception:
            continue
        manifest_path = art_dir / "workspace_patch.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
        if not isinstance(manifest, dict):
            continue
        result = load_task_result(root, child_task_id) or {}
        return art_dir / "workspace.patch", manifest, result
    listed = ", ".join(str(r) for r in roots) or "(no drive roots resolved)"
    return (
        f"⚠️ INTEGRATE_PATCH_NOT_FOUND: no workspace_patch.json for child {child_task_id!r} under {listed}. "
        "Ensure the child finished and was a mutative subagent that returned a workspace patch "
        "(retrieve it with get_task_result/wait_task first)."
    )


def _sha256_file(path: pathlib.Path) -> str:
    from hashlib import sha256

    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_verdict(
    ctx: ToolContext,
    child_task_id: str,
    *,
    outcome: str,
    reason: str,
    files: List[str],
    manifest: Dict[str, Any],
    applied: bool,
    conflicts: List[str],
    protected: List[str],
    target: str = "",
) -> str:
    parent_task_id = task_id_for_artifacts(ctx)
    art_dir = task_artifact_dir_path(getattr(ctx, "drive_root", "."), parent_task_id, create=True)
    verdict = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "tool": "integrate_subagent_patch",
        "parent_task_id": parent_task_id,
        "child_task_id": child_task_id,
        "outcome": outcome,
        "applied": bool(applied),
        "reason": str(reason or ""),
        "target_root": str(target or ""),
        "files": list(files or []),
        "protected_matches": list(protected or []),
        "conflicts": list(conflicts or []),
        "patch_sha256": str((manifest or {}).get("sha256") or ""),
        "diffstat": str((manifest or {}).get("diffstat") or ""),
    }
    path = art_dir / f"subagent_patch_verdict_{child_task_id}.json"
    try:
        atomic_write_json(path, verdict, trailing_newline=True)
    except Exception:
        return ""
    return str(path)


def _child_write_root(child_result: Dict[str, Any]) -> str:
    constraint = child_result.get("task_constraint") if isinstance(child_result.get("task_constraint"), dict) else {}
    metadata = child_result.get("metadata") if isinstance(child_result.get("metadata"), dict) else {}
    for value in (
        constraint.get("write_root"),
        child_result.get("workspace_root"),
        metadata.get("workspace_root"),
        child_result.get("write_root"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _parent_external_workspace_root(ctx: ToolContext, active_root: pathlib.Path) -> tuple[pathlib.Path | None, str]:
    """Return the parent's active external workspace root, or a fail-closed reason."""

    mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
    workspace_root = getattr(ctx, "workspace_root", None)
    if mode not in {"external", "external_workspace"} or workspace_root is None:
        return None, "parent task is not running in an external workspace mode"
    try:
        declared = pathlib.Path(workspace_root).resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        return None, f"parent workspace_root is invalid: {type(exc).__name__}: {exc}"
    resolved_active = active_root.resolve(strict=False)
    if declared != resolved_active:
        return None, (
            "parent active repo does not resolve to its declared external workspace "
            f"(active={resolved_active}, workspace_root={declared})"
        )
    return resolved_active, ""


def _verify_shared_external_workspace(
    target: pathlib.Path,
    patch_path: pathlib.Path,
    touched: List[str],
) -> tuple[bool, List[str], str]:
    invalid: List[str] = []
    resolved_target = target.resolve(strict=False)
    for rel in touched:
        text = str(rel or "").strip()
        if not text:
            continue
        path = (target / text).resolve(strict=False)
        try:
            path.relative_to(resolved_target)
        except ValueError:
            invalid.append(text)
    if invalid:
        return False, invalid, ""
    if not (target / ".git").exists():
        return False, [], f"target {target} is not a git working tree"
    proc = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=str(target),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, [], detail[:600] or "reverse patch check failed"
    return True, [], ""


def _patch_touched_paths(patch_path: pathlib.Path, target: pathlib.Path) -> tuple[set[str], str]:
    numstat = subprocess.run(
        ["git", "apply", "--numstat", str(patch_path)], cwd=str(target), capture_output=True, text=True,
    )
    if numstat.returncode != 0:
        return set(), (numstat.stderr or numstat.stdout or "").strip()[:600]
    touched: set[str] = set()
    try:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        patch_text = ""
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)\s*$", patch_text, re.MULTILINE):
        touched.add(m.group(1).strip())
        touched.add(m.group(2).strip())
    for ln in numstat.stdout.splitlines():
        if ln.strip():
            touched.add(ln.rsplit("\t", 1)[-1].strip())
    return {path for path in touched if path}, ""


def _is_host_minted_projects_tree(path: pathlib.Path) -> bool:
    """True when ``path`` is a host-minted genesis/coop tree — i.e. inside the
    durable subagent-projects root. Owner-attached folders never live there, so this
    is the structural boundary for the coop no-op (and the checkpoint-commit)."""
    try:
        from ouroboros.config import get_subagent_projects_root
        from ouroboros.tool_access import path_is_relative_to

        projects_root = pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
        return path_is_relative_to(pathlib.Path(path).resolve(strict=False), projects_root)
    except Exception:
        return False


def _maybe_coop_noop_verdict(
    ctx: ToolContext,
    *,
    child_task_id: str,
    reason: str,
    patch_path: pathlib.Path,
    manifest: Dict[str, Any],
    child_result: Dict[str, Any],
    touched: List[str],
) -> str:
    """Recognize the cooperative-build case for a NON-workspace parent and verify it
    read-only. Conditions (all structural): the child recorded a write_root that is a
    host-minted coop/genesis tree (under the subagent-projects root), the tree exists
    with a .git, and the child's patch is ALREADY IN the tree (reverse-apply check
    passes — the same check `_verify_shared_external_workspace` uses). Returns the
    successful no-op tool result, or "" when this is not the coop case (the caller
    then falls through to the parent-missing error). Never applies anything."""
    child_root = _child_write_root(child_result or {})
    if not child_root:
        return ""
    target = pathlib.Path(child_root).resolve(strict=False)
    if not _is_host_minted_projects_tree(target):
        return ""
    ok, invalid, detail = _verify_shared_external_workspace(target, patch_path, touched)
    if not ok:
        verdict_path = _write_verdict(
            ctx,
            child_task_id,
            outcome="coop_tree_verification_failed",
            reason=reason or detail or "child work not verifiable in the coop tree",
            files=touched,
            manifest=manifest,
            applied=False,
            conflicts=[detail] if detail else [f"paths escape the coop tree: {invalid[:5]}"],
            protected=[],
            target=str(target),
        )
        return (
            "⚠️ INTEGRATE_COOP_VERIFY_FAILED: child "
            f"{child_task_id} wrote to the shared coop tree {target}, but its recorded patch "
            f"does not verify against the tree ({detail or 'path escape'}). Inspect the tree "
            f"and the child result directly. Verdict: {verdict_path or '(unwritten)'}."
        )
    verdict_path = _write_verdict(
        ctx,
        child_task_id,
        outcome="coop_already_in_tree",
        reason=reason or "cooperative child wrote directly into the host-minted shared tree",
        files=touched,
        manifest=manifest,
        applied=False,
        conflicts=[],
        protected=[],
        target=str(target),
    )
    return (
        f"OK: cooperative no-op — child {child_task_id}'s work is ALREADY in the shared "
        f"coop tree {target} (verified read-only against its patch; nothing to apply). "
        f"The tree is checkpoint-committed by the host when this task tree finalizes. "
        f"Touched files: {', '.join(touched[:10]) or '(none listed)'}. "
        f"Verdict: {verdict_path or '(unwritten)'}."
    )


def _handle_external_workspace_integration(
    ctx: ToolContext,
    *,
    child_task_id: str,
    reason: str,
    requested_target: str,
    active_root: pathlib.Path,
    patch_path: pathlib.Path,
    manifest: Dict[str, Any],
    child_result: Dict[str, Any],
    touched: List[str],
) -> str:
    parent_external_root, parent_external_reason = _parent_external_workspace_root(ctx, active_root)
    if parent_external_root is None:
        # v6.58.0 (2.4A): the COOP case — a NON-workspace parent (e.g. a main-chat root)
        # whose children built in a HOST-MINTED shared coop/genesis tree. The children
        # wrote DIRECTLY into that tree, so there is nothing for the parent to apply
        # anywhere: verify read-only that the child's work is already in the tree and
        # return a SUCCESSFUL no-op verdict instead of a parent-missing error.
        coop_result = _maybe_coop_noop_verdict(
            ctx,
            child_task_id=child_task_id,
            reason=reason,
            patch_path=patch_path,
            manifest=manifest,
            child_result=child_result,
            touched=touched,
        )
        if coop_result:
            return coop_result
        verdict_path = _write_verdict(
            ctx,
            child_task_id,
            outcome="shared_workspace_parent_missing",
            reason=reason or parent_external_reason,
            files=touched,
            manifest=manifest,
            applied=False,
            conflicts=[parent_external_reason],
            protected=[],
            target=str(active_root),
        )
        return (
            "⚠️ INTEGRATE_EXTERNAL_WORKSPACE_PARENT_MISSING: external_workspace child "
            f"{child_task_id} can only be verified by a parent running in the same active "
            f"external workspace. {parent_external_reason}. Verdict: {verdict_path or '(unwritten)'}."
        )

    child_root = _child_write_root(child_result or {})
    if not child_root:
        verdict_path = _write_verdict(
            ctx,
            child_task_id,
            outcome="shared_workspace_missing_target",
            reason=reason or "child result did not record write_root/workspace_root",
            files=touched,
            manifest=manifest,
            applied=False,
            conflicts=["missing child write_root/workspace_root"],
            protected=[],
            target=str(parent_external_root),
        )
        return (
            f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISSING: child {child_task_id} did not record "
            f"the shared workspace write_root/workspace_root. Verdict: {verdict_path or '(unwritten)'}."
        )

    child_target = pathlib.Path(child_root).resolve(strict=False)
    if child_target != parent_external_root:
        verdict_path = _write_verdict(
            ctx,
            child_task_id,
            outcome="shared_workspace_target_mismatch",
            reason=reason or "child write_root/workspace_root does not match parent active external workspace",
            files=touched,
            manifest=manifest,
            applied=False,
            conflicts=[f"child={child_target}", f"parent={parent_external_root}"],
            protected=[],
            target=str(parent_external_root),
        )
        return (
            "⚠️ INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH: child wrote to "
            f"{child_target}, but this parent is active in {parent_external_root}. Do not verify or "
            "apply patches across workspaces; inspect the child result and reschedule inside the "
            f"same active workspace. Verdict: {verdict_path or '(unwritten)'}."
        )

    target = parent_external_root
    if requested_target and pathlib.Path(requested_target).resolve(strict=False) != target:
        verdict_path = _write_verdict(
            ctx,
            child_task_id,
            outcome="shared_workspace_target_mismatch",
            reason=reason or "target_root does not match parent active external workspace",
            files=touched,
            manifest=manifest,
            applied=False,
            conflicts=[f"target_root={pathlib.Path(requested_target).resolve(strict=False)}", f"parent={target}"],
            protected=[],
            target=str(target),
        )
        return (
            "⚠️ INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH: child wrote to "
            f"{child_root}, but target_root was {requested_target}. Do not verify or apply the "
            f"patch across workspaces. Verdict: {verdict_path or '(unwritten)'}."
        )

    patch_touched, parse_error = _patch_touched_paths(patch_path, target)
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse {child_task_id} workspace.patch for the "
            f"external workspace check (git apply --numstat failed): {parse_error[:300]}"
        )
    authoritative_touched = sorted(patch_touched or set(touched))
    verified, missing, mismatch_reason = _verify_shared_external_workspace(target, patch_path, authoritative_touched)
    outcome = (
        "verified_shared_workspace"
        if verified
        else ("shared_workspace_missing" if missing else "shared_workspace_mismatch")
    )
    conflicts = missing or ([mismatch_reason] if mismatch_reason else [])
    verdict_path = _write_verdict(
        ctx,
        child_task_id,
        outcome=outcome,
        reason=reason,
        files=authoritative_touched,
        manifest=manifest,
        applied=False,
        conflicts=conflicts,
        protected=[],
        target=str(target),
    )
    if verified:
        return (
            f"✅ Verified external_workspace child {child_task_id}: {len(authoritative_touched)} file(s) are already "
            f"present in the shared workspace {target}. No patch was re-applied. "
            f"Verdict: {verdict_path or '(unwritten)'}."
        )
    if missing:
        return (
            f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISSING: child {child_task_id} patch referenced "
            f"{len(missing)} invalid shared-workspace path(s) under {target}. "
            f"Paths: {missing[:20]}. Verdict: {verdict_path or '(unwritten)'}."
        )
    return (
        f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH: child {child_task_id} reported {len(authoritative_touched)} "
        f"changed file(s), but the patch does not match the current shared workspace {target}. "
        f"git said: {mismatch_reason[:600]}. Verdict: {verdict_path or '(unwritten)'}."
    )


def _integrate_subagent_patch(
    ctx: ToolContext,
    task_id: str = "",
    decision: str = "apply",
    reason: str = "",
    target_root: str = "",
) -> str:
    child_task_id = str(task_id or "").strip()
    if not child_task_id:
        return "⚠️ TOOL_ARG_ERROR (integrate_subagent_patch): task_id is required (the child whose patch to integrate)."
    decision = str(decision or "apply").strip().lower()
    if decision not in {"apply", "reject"}:
        return "⚠️ TOOL_ARG_ERROR (integrate_subagent_patch): decision must be 'apply' or 'reject'."

    located = _locate_child_patch(ctx, child_task_id)
    if isinstance(located, str):
        return located
    patch_path, manifest, child_result = located
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    touched += [str(p) for p in (manifest.get("untracked_included") or [])]

    # Top-only routing: integrate only your OWN immediate children. A descendant
    # patch must bubble up through its own parent, not jump levels into this repo.
    parent_tid = str(getattr(ctx, "task_id", "") or "").strip()
    child_parent = str((child_result or {}).get("parent_task_id") or "").strip()
    if not parent_tid:
        return (
            "⚠️ INTEGRATE_LINEAGE_FORBIDDEN: this task has no task_id, so child lineage cannot be "
            "verified. Integration is only allowed from the task whose task_id is the child's parent."
        )
    if child_parent != parent_tid:
        return (
            f"⚠️ INTEGRATE_LINEAGE_FORBIDDEN: {child_task_id} is not a direct child of this task "
            f"(its parent is {child_parent or '(unknown)'!r}, not {parent_tid!r}). Top-only routing: "
            "integrate only your own immediate children; descendant patches bubble up one parent at a time."
        )

    # genesis projects are standalone deliverables (the project directory itself),
    # NOT live-body patches. Machine-enforce the documented invariant that a genesis
    # child is never integrated into the active repo, regardless of decision=apply.
    child_surface = str(((child_result or {}).get("task_constraint") or {}).get("surface") or "")
    if child_surface == "genesis" and decision != "reject":
        return (
            f"⚠️ INTEGRATE_GENESIS_FORBIDDEN: {child_task_id} is a from-scratch (genesis) project; "
            "its deliverable is the project directory itself, not a patch for this repo. Do not integrate "
            "it into the live body — use the project at its write_root directly (or decision='reject' to "
            "record a verdict)."
        )

    if decision == "reject":
        verdict_path = _write_verdict(
            ctx, child_task_id, outcome="rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[],
        )
        return (
            f"🚫 Rejected subagent patch from {child_task_id} ({len(touched)} file(s) not applied). "
            f"Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}."
        )

    status = str(manifest.get("status") or "")
    if status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        return (
            f"⚠️ INTEGRATE_NO_CHANGES: child {child_task_id} workspace patch status={status!r}; "
            "nothing to apply."
        )
    if not patch_path.exists():
        return f"⚠️ INTEGRATE_PATCH_MISSING: workspace.patch for {child_task_id} not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if expected_digest:
        actual_digest = _sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for {child_task_id} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )

    # Top-only routing for EVERY caller: integration always targets your OWN active
    # repo/worktree. An explicit target_root must equal it (no foreign target, which
    # could be the live repo or another worktree).
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    requested_target = str(target_root or "").strip()
    if (
        requested_target
        and child_surface != "external_workspace"
        and pathlib.Path(requested_target).resolve(strict=False) != active_root
    ):
        return (
            "⚠️ INTEGRATE_TARGET_FORBIDDEN: integration targets only your own active repo/worktree "
            "(top-only routing). Drop target_root or set it to the active root; descendant patches "
            "bubble up one parent at a time."
        )
    target = active_root
    if not (target / ".git").exists():
        if child_surface != "external_workspace":
            return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."

    if child_surface == "external_workspace":
        return _handle_external_workspace_integration(
            ctx,
            child_task_id=child_task_id,
            reason=reason,
            requested_target=requested_target,
            active_root=active_root,
            patch_path=patch_path,
            manifest=manifest,
            child_result=child_result,
            touched=touched,
        )

    # Fail-closed category guard (v6.56.0): a self_worktree child's patch is a
    # patch AGAINST THE OUROBOROS SYSTEM REPO. A parent running in EXTERNAL
    # workspace mode has the external project as its active root — applying a
    # system-repo patch there would target the wrong repository. Refuse instead
    # of 3-way-applying into the task workspace. A nested acting parent whose
    # own workspace IS a self_worktree checkout stays legitimate top-only
    # routing and is not touched by this guard.
    if child_surface == "self_worktree":
        parent_ws_mode = str(getattr(ctx, "workspace_mode", "") or "").strip().lower()
        # Fire STRUCTURALLY whenever the parent's active root is a non-system
        # workspace (is_workspace_mode()), so an unrecognized external spelling
        # cannot slip past a fixed allowlist. The one excluded mode is a parent
        # whose OWN workspace is a self_worktree checkout — it legitimately routes
        # a system-repo patch (nested acting), as the comment above notes.
        if ctx.is_workspace_mode() and parent_ws_mode != "self_worktree":
            return (
                f"⚠️ INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE: child {child_task_id} produced a "
                "self_worktree patch (against the Ouroboros system repo), but this task's active "
                "root is an external workspace. Refusing to apply a system-repo patch into the "
                "task workspace; integrate it from a non-workspace parent task instead."
            )

    runtime_mode = get_runtime_mode()
    # Derive the changed-path set from the PATCH ITSELF (not the child-controlled
    # manifest) for the protected-path gate: a child must not be able to hide a
    # protected edit by omitting it from the manifest (sha256 verifies bytes only).
    patch_touched, parse_error = _patch_touched_paths(patch_path, target)
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse {child_task_id} workspace.patch for the "
            f"protected-path check (git apply --numstat failed): {parse_error[:300]}"
        )
    protected = protected_paths_in(sorted(patch_touched))
    if protected:
        grant_ok = (not is_acting) or bool(getattr(constraint, "protected_paths_grant", False))
        if not (mode_allows_protected_write(runtime_mode) and grant_ok):
            _write_verdict(
                ctx, child_task_id, outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[], protected=[p.path for p in protected],
                target=str(target),
            )
            return protected_write_block_message(
                path=protected[0].path,
                runtime_mode=runtime_mode,
                action=f"integrate subagent patch {child_task_id} touching",
            )

    # Serialize the index/worktree mutation with the SAME repo git lock that
    # commit_reviewed uses, so a concurrent integration or a reviewed commit cannot
    # race on the index.
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    try:
        _git_lock = _acquire_git_lock(ctx)
    except Exception as exc:
        return f"⚠️ INTEGRATE_LOCK_TIMEOUT: could not acquire the repo git lock: {type(exc).__name__}: {exc}."
    try:
        proc = subprocess.run(
            ["git", "apply", "--3way", "--index", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
    finally:
        _release_git_lock(_git_lock)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        conflicts = [ln.strip() for ln in stderr.splitlines() if "conflict" in ln.lower() or "patch failed" in ln.lower()]
        _write_verdict(
            ctx, child_task_id, outcome="conflict", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=conflicts or [stderr[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        return (
            f"⚠️ INTEGRATE_CONFLICT: 3-way apply of {child_task_id} into {target} did not apply cleanly. "
            f"git said: {stderr[:600]}\n"
            "Inspect with vcs_diff and resolve, or run vcs_restore to abort, then retry or pick another child."
        )

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_subagent_patch",
        )
    except Exception:
        pass

    verdict_path = _write_verdict(
        ctx, child_task_id, outcome="applied", reason=reason, files=touched,
        manifest=manifest, applied=True, conflicts=[], protected=[p.path for p in protected],
        target=str(target),
    )
    diffstat = str(manifest.get("diffstat") or "").strip()
    note = ""
    if protected:
        note = f" Includes {len(protected)} protected path(s) (allowed: runtime_mode={runtime_mode})."
    return (
        f"✅ Integrated subagent patch from {child_task_id} into {target} ({len(touched)} file(s), staged).{note}\n"
        f"{diffstat}\n"
        f"Verdict: {verdict_path or '(unwritten)'}.\n"
        "Changes are staged but NOT committed — review and run commit_reviewed yourself (you are the sole committer)."
    )


# Per-candidate diff preview cap. Kept well under the tool's 80_000-char result
# limit (tool_capabilities.TOOL_RESULT_LIMITS) so several candidates fit side by
# side without the outer truncation hiding later candidates.
_COMPARE_PATCH_PREVIEW_CHARS = 12000


def _compare_subagent_patches(ctx: ToolContext, task_ids: Any = None) -> str:
    """Read-only best-of-N helper: show several children's returned patches side by
    side so the parent can synthesize LLM-first. Applies/commits nothing."""
    if isinstance(task_ids, str):
        ids = [task_ids.strip()] if task_ids.strip() else []
    else:
        ids = [str(t).strip() for t in (task_ids or []) if str(t).strip()]
    if not ids:
        return (
            "⚠️ TOOL_ARG_ERROR (compare_subagent_patches): task_ids must be a non-empty list of "
            "child subagent task_ids (the candidates to compare)."
        )
    parts: List[str] = [f"# Candidate comparison — {len(ids)} subagent patch(es)"]
    for cid in ids:
        located = _locate_child_patch(ctx, cid)
        if isinstance(located, str):
            parts.append(f"\n## {cid}\n{located}")
            continue
        patch_path, manifest, child_result = located
        status = str(manifest.get("status") or "")
        diffstat = str(manifest.get("diffstat") or "").strip()
        tracked = [str(p) for p in (manifest.get("tracked_changed") or [])]
        untracked = [str(p) for p in (manifest.get("untracked_included") or [])]
        result_status = str((child_result or {}).get("status") or "")
        result_summary = str((child_result or {}).get("result") or "").strip()
        if len(result_summary) > 600:
            result_summary = result_summary[:600] + " …"
        body = ""
        if patch_path.exists():
            try:
                raw = patch_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            if len(raw) > _COMPARE_PATCH_PREVIEW_CHARS:
                body = raw[:_COMPARE_PATCH_PREVIEW_CHARS] + (
                    f"\n... [patch preview truncated; {len(raw)} bytes total — "
                    "integrate to apply, or read the workspace.patch artifact for the full diff] ..."
                )
            else:
                body = raw
        parts.append(
            f"\n## {cid}\n"
            f"- patch status: {status or '(none)'} | child result status: {result_status or '(unknown)'}\n"
            f"- tracked changed: {len(tracked)} | untracked included: {len(untracked)}\n"
            f"- diffstat: {diffstat or '(none)'}\n"
            + (f"- child summary: {result_summary}\n" if result_summary else "")
            + (f"\n```diff\n{body}\n```\n" if body else "- (no patch body; nothing to apply)\n")
        )
    parts.append(
        "\nPick the best candidate and apply it with integrate_subagent_patch(task_id=...), "
        "or synthesize across candidates yourself (you are the sole committer). Comparison is read-only."
    )
    return "\n".join(parts)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "compare_subagent_patches",
            {
                "name": "compare_subagent_patches",
                "description": (
                    "Read-only best-of-N helper: show several mutative children's returned "
                    "workspace.patch candidates side by side (status, diffstat, changed-file counts, "
                    "child summary, and a bounded diff preview) so you can pick the best one or "
                    "synthesize across them. Applies and commits NOTHING — use integrate_subagent_patch "
                    "to actually stage a chosen patch. Only sees patches reachable from your task drive roots."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Child subagent task_ids of the candidates to compare.",
                        },
                    },
                    "required": ["task_ids"],
                },
            },
            _compare_subagent_patches,
        ),
        ToolEntry(
            "integrate_subagent_patch",
            {
                "name": "integrate_subagent_patch",
                "description": (
                    "Integrate (apply) a mutative subagent's returned workspace.patch into your active "
                    "repo/worktree, or record a rejection. You remain the SOLE COMMITTER: this stages the "
                    "child's changes (manifest-first, sha256-verified, 3-way apply) but does NOT commit — "
                    "you review and run commit_reviewed yourself. Use for best-of-N: pick the best child "
                    "and integrate it, or integrate several to synthesize. Protected-path changes require "
                    "pro runtime mode (and, for a nested acting parent, protected_paths_grant). Conflicts "
                    "are reported for you to resolve (vcs_diff) or abort (vcs_restore). Writes a "
                    "subagent_patch_verdict_<task_id>.json audit artifact."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The child subagent task_id whose workspace.patch to integrate."},
                        "decision": {"type": "string", "enum": ["apply", "reject"], "default": "apply", "description": "apply = stage the child's patch; reject = record a rejection verdict without applying."},
                        "reason": {"type": "string", "description": "Optional rationale recorded in the verdict (why accept / reject / synthesize)."},
                        "target_root": {"type": "string", "description": "Optional explicit target repo/worktree root. Defaults to your active repo (live repo for the root agent; your worktree for a nested acting parent — top-only routing)."},
                    },
                    "required": ["task_id"],
                },
            },
            _integrate_subagent_patch,
        ),
    ]
