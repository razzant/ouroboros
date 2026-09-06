"""``integrate_subagent_patch``: the parent's manifest-first integration tool.

A mutative (acting) subagent returns its changes as a ``workspace.patch`` artifact
(produced by headless finalization, a git diff against the child's base commit).
The parent decides what to do with it — accept one (best-of-N), synthesize several,
or reject. This tool applies isolated self_worktree patches to the parent's repo,
or verifies native external_workspace changes already present in the shared tree.
The parent stays the sole committer: applying stages changes but never
commits; the parent reviews and runs ``commit_reviewed`` itself.

Routing is top-only: ``target_root`` defaults to ``ctx.active_repo_dir()`` — the
live repo for the root agent, or the parent's own worktree for a nested acting
parent, so descendants bubble their patches up one level at a time.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple, Union

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.artifacts import task_artifact_dir_path
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
from ouroboros.headless import (
    ARTIFACT_STATUS_READY_NO_CHANGES,  # noqa: F401
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)

log = logging.getLogger(__name__)

# The capture statuses a disposition may proceed over (C1-R3): a usable patch
# exists (with changes), or the run provably changed nothing. Everything else —
# failed, missing, unreadable — must never be applied over or release a snapshot.


def _record_integration_disposition(
    ctx: ToolContext,
    child_task_id: str,
    disposition: str,
    reason: str,
    default_reason: str,
) -> str:
    """Stamp only a genuinely completed apply/verify/reject operation."""
    from ouroboros.tools.join_ledger import _record_current_child_result_disposition
    recorded = _record_current_child_result_disposition(
        ctx,
        child_task_id,
        disposition,
        reason or default_reason,
    )
    if recorded.startswith("OK:"):
        return ""
    return f"\n⚠️ INTEGRATE_DISPOSITION_FAILED: {recorded}"


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


# Verdict writing lives in tools/patch_verdict.py (module-size ceiling);
# the historical private name stays importable for callers and tests.
from ouroboros.tools.patch_verdict import write_patch_verdict as _write_verdict  # noqa: E402


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


def _patch_touched_paths(patch_path: pathlib.Path, target: pathlib.Path, env: Any = None) -> tuple[set[str], str]:
    """Every path the patch touches, parsed NUL-SAFELY from git's own reader.

    ``git apply --numstat -z`` is the machine-readable form: fields are
    NUL-terminated and pathnames are never munged (no quoting, no ``\\t``/``\\n``
    escapes). The previous text parse split on tabs and additionally regex-scanned
    ``diff --git a/… b/…`` headers, so a path containing a tab, a newline or a
    quote-triggering byte produced a corrupted pathspec — which then reached the
    protected-path gate and the staging step.

    Read in BOTH directions, because ``git apply --numstat`` names only the paths
    that direction WRITES: a rename reports its destination forward and its source
    in reverse, and the source is exactly the deletion the staging step must
    record. (This is where the regex scan earned its keep; the union of the two
    NUL-safe readings replaces it without the quoting bug.)
    """
    touched: set[str] = set()
    for direction in ([], ["-R"]):
        numstat = subprocess.run(
            ["git", "apply", *direction, "--numstat", "-z", str(patch_path)],
            cwd=str(target), capture_output=True, env=env,
        )
        if numstat.returncode != 0:
            detail = (numstat.stderr or numstat.stdout or b"").decode("utf-8", errors="replace")
            return set(), detail.strip()[:600]
        tokens = (numstat.stdout or b"").split(b"\0")
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if not token:
                continue
            parts = token.split(b"\t", 2)
            if len(parts) < 3:
                continue
            path = parts[2]
            if path:
                touched.add(path.decode("utf-8", errors="surrogateescape"))
                continue
            # `git diff --numstat -z` spells a rename as an empty path field
            # followed by two more NUL-terminated fields (source, destination).
            for _ in range(2):
                if index < len(tokens):
                    extra = tokens[index]
                    index += 1
                    if extra:
                        touched.add(extra.decode("utf-8", errors="surrogateescape"))
    return {path for path in touched if path}, ""


def _stageable_paths(target: pathlib.Path, touched: List[str]) -> List[str]:
    """The subset of ``touched`` git can actually stage after a successful apply.

    A path is stageable when it EXISTS on disk (added or modified) or is in the
    index (then it stages as a deletion). A path that is neither — an UNTRACKED
    file the patch deleted, which the C1 baseline carried as a tree entry but the
    target never tracked — has nothing to stage, and naming it made ``git add``
    exit non-zero ("did not match any files") AFTER the apply had already mutated
    the tree.
    """
    present = {p for p in touched if os.path.lexists(str(target / p))}
    missing = [p for p in touched if p not in present]
    if not missing:
        return sorted(present)
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=str(target), capture_output=True)
    indexed: set[str] = set()
    if listed.returncode == 0:
        indexed = {
            chunk.decode("utf-8", errors="surrogateescape")
            for chunk in (listed.stdout or b"").split(b"\0") if chunk
        }
    return sorted(present | {p for p in missing if p in indexed})


def _baseline_drifted_paths(
    target: pathlib.Path, baseline_sha: str, touched: List[str],
) -> tuple[List[str], str]:
    """Touched paths whose CURRENT target state differs from the run's baseline.

    A plain ``git apply`` relocates hunks by offset and ignores whole files whose
    context happens to still match, so "the target drifted since the snapshot"
    cannot be inferred from the apply's exit code — a moved target could be
    patched at a shifted position and silently accepted. Drift is therefore
    PROVEN first, against the baseline commit, with the same temp-index machinery
    the baseline itself was built with (identical filter/attribute treatment):
    seed a scratch index from the baseline tree, stage the current worktree state
    of exactly these paths into it, and ask git which entries now differ.

    Returns ``(drifted_paths, error)``; a non-empty error means the comparison
    could not be made and the caller must refuse rather than apply blind.
    """
    if not touched:
        return [], ""
    if not str(baseline_sha or "").strip():
        return [], "the run's custody row carries no baseline commit"
    payload = b"\0".join(p.encode("utf-8", errors="surrogateescape") for p in touched) + b"\0"
    scratch = tempfile.mkdtemp(prefix="ouro_baseline_drift_")
    env = {**os.environ, "GIT_INDEX_FILE": str(pathlib.Path(scratch) / "index")}
    try:
        for args in (
            ["git", "read-tree", str(baseline_sha)],
            ["git", "update-index", "-z", "--add", "--remove", "--stdin"],
            ["git", "diff-index", "--cached", "--name-only", "-z", str(baseline_sha)],
        ):
            proc = subprocess.run(
                args, cwd=str(target), capture_output=True, env=env,
                input=payload if args[1] == "update-index" else None,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
                return [], f"{args[1]} failed: {detail.strip()[:300]}"
        drifted = sorted(
            chunk.decode("utf-8", errors="surrogateescape")
            for chunk in (proc.stdout or b"").split(b"\0") if chunk
        )
        return drifted, ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _target_is_system_repo(ctx: ToolContext) -> bool:
    """Whether an integration target is the OUROBOROS body (or a checkout of it).

    The protected-path policy (`BIBLE.md`, `.github/workflows/ci.yml`, `build.sh`,
    `ouroboros/contracts/…`) is about THIS repository's own invariants. A foreign
    project that happens to own files with those names is not covered by it, and
    gating there blocks ordinary work in an external workspace with advice about a
    runtime mode that has nothing to do with that project. The predicate mirrors
    the registry's own write gate (`(not workspace_mode or acting_self_worktree)`):
    no active workspace means the active root IS the live repo, and a
    ``self_worktree`` surface is a checkout of it.
    """
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if str(getattr(constraint, "surface", "") or "") == "self_worktree":
        return True
    if str(getattr(ctx, "workspace_mode", "") or "").strip().lower() == "self_worktree":
        return True
    try:
        return not bool(ctx.is_workspace_mode())
    except Exception:
        return True


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
    disposition_warning = _record_integration_disposition(
        ctx,
        child_task_id,
        "integrated",
        reason,
        "verified that the child result is already integrated in the cooperative tree",
    )
    return (
        f"OK: cooperative no-op — child {child_task_id}'s work is ALREADY in the shared "
        f"coop tree {target} (verified read-only against its patch; nothing to apply). "
        f"The tree is checkpoint-committed by the host when this task tree finalizes. "
        f"Touched files: {', '.join(touched[:10]) or '(none listed)'}. "
        f"Verdict: {verdict_path or '(unwritten)'}.{_format_patch_exclusions(manifest)}{disposition_warning}"
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

    def _target_mismatch_verdict(why: str, conflicts: List[str], root: Any) -> str:
        return _write_verdict(
            ctx, child_task_id, outcome="shared_workspace_target_mismatch",
            reason=reason or why, files=touched, manifest=manifest, applied=False,
            conflicts=conflicts, protected=[], target=str(root))

    child_target = pathlib.Path(child_root).resolve(strict=False)
    if child_target != parent_external_root:
        verdict_path = _target_mismatch_verdict(
            "child write_root/workspace_root does not match parent active external workspace",
            [f"child={child_target}", f"parent={parent_external_root}"],
            parent_external_root)
        return (
            "⚠️ INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH: child wrote to "
            f"{child_target}, but this parent is active in {parent_external_root}. Do not verify or "
            "apply patches across workspaces; inspect the child result and reschedule inside the "
            f"same active workspace. Verdict: {verdict_path or '(unwritten)'}."
        )

    target = parent_external_root
    if requested_target and pathlib.Path(requested_target).resolve(strict=False) != target:
        verdict_path = _target_mismatch_verdict(
            "target_root does not match parent active external workspace",
            [f"target_root={pathlib.Path(requested_target).resolve(strict=False)}",
             f"parent={target}"],
            target)
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
        disposition_warning = _record_integration_disposition(
            ctx,
            child_task_id,
            "integrated",
            reason,
            "verified that the child result is already integrated in the shared external workspace",
        )
        return (
            f"✅ Verified external_workspace child {child_task_id}: {len(authoritative_touched)} file(s) are already "
            f"present in the shared workspace {target}. No patch was re-applied. "
            f"Verdict: {verdict_path or '(unwritten)'}.{_format_patch_exclusions(manifest)}{disposition_warning}"
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
        disposition_warning = _record_integration_disposition(
            ctx,
            child_task_id,
            "irrelevant",
            reason,
            "rejected the child result after review",
        )
        return (
            f"🚫 Rejected subagent patch from {child_task_id} ({len(touched)} file(s) not applied). "
            f"Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}."
            f"{_format_patch_exclusions(manifest)}{disposition_warning}"
        )

    status = str(manifest.get("status") or "")
    if status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        return (
            f"⚠️ INTEGRATE_NO_CHANGES: child {child_task_id} workspace patch status={status!r}; "
            "nothing to apply."
            f"{_format_patch_exclusions(manifest)}"
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
    disposition_warning = _record_integration_disposition(
        ctx,
        child_task_id,
        "integrated",
        reason,
        "applied and staged the child result in the parent worktree",
    )
    return (
        f"✅ Integrated subagent patch from {child_task_id} into {target} ({len(touched)} file(s), staged).{note}\n"
        f"{diffstat}{_format_patch_exclusions(manifest)}\n"
        f"Verdict: {verdict_path or '(unwritten)'}.\n"
        "Changes are staged but NOT committed — review and run commit_reviewed yourself (you are the sole committer)."
        f"{disposition_warning}"
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
                    "integrate to apply/verify, or read the workspace.patch artifact for the full diff] ..."
                )
            else:
                body = raw
        parts.append(
            f"\n## {cid}\n"
            f"- patch status: {status or '(none)'} | child result status: {result_status or '(unknown)'}\n"
            f"- tracked changed: {len(tracked)} | untracked included: {len(untracked)}\n"
            f"- diffstat: {diffstat or '(none)'}{_format_patch_exclusions(manifest)}\n"
            + (f"- child summary: {result_summary}\n" if result_summary else "")
            + (f"\n```diff\n{body}\n```\n" if body else "- (no patch body; nothing to apply)\n")
        )
    parts.append(
        "\nUse integrate_subagent_patch(task_id=...) to apply an isolated patch or verify shared files, "
        "or synthesize across candidates yourself (you are the sole committer). Comparison is read-only."
    )
    return "\n".join(parts)


from ouroboros.workspace_patch_rules import (  # noqa: E402 — patch-rule SSOT
    format_patch_exclusions as _format_patch_exclusions,
)


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
                    "to apply an isolated patch or verify shared files. Only sees patches reachable from your task drive roots."
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
                    "Integrate a mutative child's result or record a rejection: self_worktree uses "
                    "manifest-first, sha256-verified 3-way apply into your active repo; native external_workspace "
                    "verifies files already in the shared tree WITHOUT reapplying. Genesis is a standalone "
                    "directory, not a repo patch. This never commits; self-modification still requires your "
                    "commit_reviewed. For best-of-N pick a child "
                    "and integrate it, or integrate several to synthesize. Protected-path changes require "
                    "pro runtime mode (and, for a nested acting parent, protected_paths_grant). Conflicts "
                    "are reported for you to resolve (vcs_diff) or abort (vcs_restore). Writes a "
                    "subagent_patch_verdict_<task_id>.json audit artifact."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The child subagent task_id whose workspace.patch to integrate."},
                        "decision": {"type": "string", "enum": ["apply", "reject"], "default": "apply", "description": "apply = apply/stage an isolated self_worktree patch, or verify shared external_workspace files already written; reject = record a rejection without applying."},
                        "reason": {"type": "string", "description": "Optional rationale recorded in the verdict (why accept / reject / synthesize)."},
                        "target_root": {"type": "string", "description": "Optional explicit target repo/worktree root. Defaults to your active repo (live repo for the root agent; your worktree for a nested acting parent — top-only routing)."},
                    },
                    "required": ["task_id"],
                },
            },
            _integrate_subagent_patch,
        ),
        ToolEntry(
            "integrate_delegated_patch",
            {
                "name": "integrate_delegated_patch",
                "description": (
                    "EXPLICITLY apply or reject the captured patch of ONE of your own delegated runs (delegate_start), or a terminal owner's orphan. Applying requires the caller's active Git root or fresh payload binding to equal the run's recorded target. Rejecting a terminal-owner orphan requires only the owner's terminality; it exists to release a dead task's locks and snapshot. A mutating delegated run edits a PRIVATE execution "
                    "snapshot; its diff is captured at terminal, and NOTHING reaches your tree "
                    "until you call this. apply = stage the run's diff into your active root "
                    "(sha256-verified; under the repo git lock every touched path is first "
                    "compared against the run's baseline, then applied to the working tree and "
                    "staged; protected paths are gated only when the target IS the Ouroboros "
                    "repo) — staged, never committed; you remain the sole committer. reject = "
                    "record a rejection and discard. Either DURABLY RECORDED disposition "
                    "releases the run's execution snapshot; a CONFLICT (a path drifted since "
                    "the snapshot) keeps snapshot and patch as resolution material you own. "
                    "For a skill-payload run (delegate_start root='skill_payload') apply is "
                    "instead a LIVE apply into the non-Git payload, guarded by a whole-payload "
                    "content-hash CAS — nothing is staged into your active root — and the "
                    "skill's existing review goes STALE: it must be re-run before the skill "
                    "is relied on. Read the captured diff (see delegate_wait's "
                    "workspace_capture block) before applying — the run's output is a claim, "
                    "not a verified result. Finalizing your task while one of your runs is neither "
                    "applied nor rejected leaves your custody audit unreconciled: the task completes as "
                    "Done with warnings (reason delegated_custody_unreconciled); reject is the closing move."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The delegated run whose captured patch to integrate (from delegate_start)."},
                        "decision": {"type": "string", "enum": ["apply", "reject"], "default": "apply", "description": "apply = integrate the run's captured diff (Git targets: applied and STAGED into your active root; skill-payload runs: applied LIVE into the non-Git payload under the content-hash CAS, nothing staged anywhere); reject = record a rejection and release the snapshot."},
                        "reason": {"type": "string", "description": "Optional rationale recorded in the verdict and the durable disposition row."},
                        "acknowledge_ambiguous": {"type": "boolean", "default": False, "description": "Set true ONLY after inspecting an INTEGRATE_DELEGATED_APPLY_AMBIGUOUS state (a crashed apply left a durable unresolved intent): resolves that stale intent and re-runs the normal disposition guards, which re-verify the tree. A no-op when no ambiguity is pending."},
                    },
                    "required": ["run_id"],
                },
            },
            lambda ctx, run_id="", decision="apply", reason="", acknowledge_ambiguous=False: _integrate_delegated_patch(
                ctx, run_id, decision, reason, acknowledge_ambiguous=bool(acknowledge_ambiguous)),
        ),
    ]


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.subagent_integration_delegated import (  # noqa: E402, F401 -- intentional public re-exports
    _READY_CAPTURE_STATUSES,
    _capture_at_disposition,
    _capture_failed_refusal,
    _delegated_disposition_refusal,
    _dispose_delegated,
    _drift_refusal,
    _integrate_delegated_patch,
    _locked_apply,
    _manifest_capture_status,
    _resolve_acknowledged_intent,
    _unwritten_disposition_text,
)
