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
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)

log = logging.getLogger(__name__)

# The capture statuses a disposition may proceed over (C1-R3): a usable patch
# exists (with changes), or the run provably changed nothing. Everything else —
# failed, missing, unreadable — must never be applied over or release a snapshot.
_READY_CAPTURE_STATUSES = frozenset({
    ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES})


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


def _patch_touched_paths(patch_path: pathlib.Path, target: pathlib.Path) -> tuple[set[str], str]:
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
            cwd=str(target), capture_output=True,
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


def _drift_refusal(
    ctx: ToolContext, *, rid: str, reason: str, touched: List[str],
    manifest: Dict[str, Any], protected: List[Any], target: pathlib.Path,
    drifted: List[str], drift_error: str,
) -> str:
    """The typed, nanny-owned refusal for a target that moved (or could not be
    compared) since the run's snapshot. Nothing was applied; material persists."""
    verdict_path = _write_verdict(
        ctx, f"run_{rid}", outcome="baseline_drift" if drifted else "baseline_unverifiable",
        reason=reason, files=touched, manifest=manifest, applied=False,
        conflicts=(drifted[:50] if drifted else [drift_error]),
        protected=[p.path for p in protected], target=str(target),
    )
    if drift_error:
        return (
            f"⚠️ INTEGRATE_DELEGATED_BASELINE_UNVERIFIABLE: run {rid}'s patch was NOT "
            f"applied — its target state could not be compared against the run's "
            f"baseline ({drift_error}). Nothing was changed; the execution snapshot "
            f"and the patch are preserved. Verdict: {verdict_path or '(unwritten)'}."
        )
    return (
        f"⚠️ INTEGRATE_CONFLICT: {len(drifted)} path(s) in {target} CHANGED since run "
        f"{rid}'s snapshot was taken, so its patch was NOT applied (a plain apply "
        f"would relocate hunks and silently land them on moved content). Drifted: "
        f"{', '.join(drifted[:10])}{' …' if len(drifted) > 10 else ''}\n"
        "YOU own this conflict: the execution snapshot and the captured patch are "
        "preserved until you resolve it. Reconcile your tree with the snapshot's "
        "changes (read the patch artifact, or diff against the execution root "
        "directly), then retry the apply, or "
        "integrate_delegated_patch(decision='reject') to discard. "
        f"Verdict: {verdict_path or '(unwritten)'}."
    )


def _locked_apply(
    ctx: ToolContext, target: pathlib.Path, patch_path: pathlib.Path,
    ordered_touched: List[str], baseline_sha: str,
) -> Dict[str, Any]:
    """Apply one captured patch under the repo git lock — mechanics only.

    Returns the FACTS the caller turns into a verdict: ``proc`` (None when
    nothing was attempted), ``drifted``/``drift_error`` from the pre-apply
    baseline comparison, and ``staging_failure``/``reverted`` for an apply whose
    staging failed. No verdicts, no dispositions, no messages — those belong to
    the one caller so every exit stays visible in one place.
    """
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    result: Dict[str, Any] = {
        "proc": None, "drifted": [], "drift_error": "",
        "staging_failure": "", "reverted": False, "lock_error": "",
    }
    try:
        _git_lock = _acquire_git_lock(ctx)
    except Exception as exc:
        result["lock_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        # DRIFT IS PROVEN, NOT INFERRED. `git apply` relocates hunks by offset, so a
        # target that moved since the snapshot can still take the patch — at a
        # shifted position, silently. Under the same lock that serializes the
        # mutation, every touched path is compared against the run's baseline commit
        # first; ANY difference is the typed conflict the nanny owns, and nothing is
        # applied.
        try:
            result["drifted"], result["drift_error"] = _baseline_drifted_paths(
                target, baseline_sha, ordered_touched)
        except Exception as exc:
            result["drifted"], result["drift_error"] = [], f"{type(exc).__name__}: {exc}"
        if result["drift_error"] or result["drifted"]:
            return result
        # WORKING-TREE apply, not --3way/--index: the baseline deliberately
        # snapshots the target's DIRTY state (that is the whole point of C1), so the
        # patch's preimage is the live working tree — while `--3way` implies index
        # binding and refuses any file whose worktree differs from the index, i.e.
        # refuses the normal shared-tree state. Touched paths are then staged
        # explicitly so the result matches integrate_subagent_patch's staged contract.
        proc = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
        result["proc"] = proc
        stageable = _stageable_paths(target, ordered_touched) if proc.returncode == 0 else []
        if proc.returncode != 0 or not stageable:
            return result
        # NUL-delimited stdin pathspecs: byte-safe and immune to argv limits for a
        # patch touching thousands of files.
        add = subprocess.run(
            ["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"],
            cwd=str(target), capture_output=True,
            input=b"\0".join(
                p.encode("utf-8", errors="surrogateescape") for p in stageable) + b"\0",
        )
        if add.returncode == 0:
            return result
        # The APPLY SUCCEEDED — the tree is already mutated. Reporting this as a
        # conflict ("the tree moved") invited a retry over changed content and let a
        # later reject record "not applied" over real changes. Try to put the tree
        # back cleanly; whatever the outcome, the caller says exactly what is true.
        result["staging_failure"] = (add.stderr or add.stdout or b"").decode(
            "utf-8", errors="replace").strip()
        check = subprocess.run(
            ["git", "apply", "--check", "--reverse", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
        if check.returncode == 0:
            result["reverted"] = subprocess.run(
                ["git", "apply", "--reverse", str(patch_path)],
                cwd=str(target), capture_output=True, text=True,
            ).returncode == 0
        return result
    finally:
        _release_git_lock(_git_lock)


def _manifest_capture_status(manifest_path: pathlib.Path) -> str:
    """The status one capture manifest reports about ITSELF; "" when unreadable."""
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    return str(loaded.get("status") or "") if isinstance(loaded, dict) else ""


def _capture_failed_refusal(rid: str, cap_status: str, note: str) -> str:
    """The ONE typed refusal for disposing over a non-usable capture (C1-R3).
    Shared by the capture-at-disposition seam and the reject branch's own guard
    so they cannot drift; no caller records a disposition over this message."""
    return (
        f"⚠️ INTEGRATE_DELEGATED_CAPTURE_FAILED: run {rid}'s changes were never "
        f"usably captured, and the capture-at-disposition attempt did not "
        f"produce a patch (status {cap_status or 'missing'!r}: {note[:300]}). "
        "No disposition was recorded — the obligation stays open and the "
        "execution snapshot (if present) is preserved. Inspect the snapshot "
        "directly, then retry this call."
    )


def _capture_at_disposition(
    drive: Any, entry: Any, rid: str, manifest_path: pathlib.Path,
) -> str:
    """Capture-on-demand (C1-R2) for a run that settled without terminal proof.

    A run reconciled while its daemon was absent or unreadable settled WITHOUT a
    capture — its state was unknowable then, so nothing was frozen. By disposition
    time the stale child is as settled as it will ever be, so this is the honest
    latest-possible capture point, through the SAME drive-rooted core the sweep
    and the nanny use. Returns "" when a usable capture exists (pre-existing or
    fresh); a capture that fails here is a typed refusal, because disposing over
    nothing would silently discard (reject) or skip (apply) work that still sits
    in the snapshot.

    The early return trusts ``patch_captured`` only together with the manifest's
    OWN ready status (C1-R3): a durable row minted by pre-R3 code over a failed
    manifest replays as not-captured, and the core is asked to re-capture. An
    exception ESCAPING the core (its internal try covers only the diff itself)
    is the same typed refusal, never a raw traceback out of the tool.
    """
    if entry.patch_captured and _manifest_capture_status(manifest_path) in _READY_CAPTURE_STATUSES:
        return ""
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    try:
        block = capture_terminal_patch_for_drive(drive, entry) or {}
    except Exception as exc:
        log.warning("capture-at-disposition raised for run %s", rid, exc_info=True)
        block = {"status": "", "note": f"capture raised {type(exc).__name__}: {exc}"}
    cap_status = str(block.get("status") or "")
    if cap_status in _READY_CAPTURE_STATUSES:
        return ""
    return _capture_failed_refusal(rid, cap_status, str(block.get("note") or ""))


def _delegated_disposition_refusal(status: str, entry: Any, rid: str,
                                   acknowledge_ambiguous: bool = False) -> str:
    """The early typed refusals of `integrate_delegated_patch`, one author.

    Ownership, isolation, finality, terminality, and the CR1-3 apply-intent
    ambiguity — every answer that needs no capture and mutates nothing.
    Returns "" when the disposition may proceed. ``acknowledge_ambiguous``
    (CR2-1) waives ONLY the ambiguity refusal: the caller then resolves the
    stale intent durably and runs the normal disposition guards from scratch.
    """
    from ouroboros import delegate_custody as custody

    if status != custody.OWNED or entry is None:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_OWNED: run {rid!r} is {status} to this task. "
            "Only the task that started a delegated run may integrate its patch."
        )
    if not entry.execution_root:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_ISOLATED: run {rid} recorded no execution "
            "snapshot (read-only, or a pre-isolation run). There is no captured patch "
            "to integrate."
        )
    if entry.patch_disposed:
        return (
            f"⚠️ INTEGRATE_DELEGATED_ALREADY_DISPOSED: run {rid}'s patch was already "
            f"{entry.patch_disposed}. A disposition is final; nothing was changed."
        )
    if not entry.settled:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_TERMINAL: run {rid} has not settled yet. "
            "delegate_wait it to terminal first — its patch is captured there."
        )
    if entry.patch_apply_pending and not acknowledge_ambiguous:
        # CR1-3: a durable apply-intent row with no resolution/disposition — a
        # previous process started the apply and died. The target MAY carry the
        # patch, so BOTH decisions refuse (a reject would record "not applied"
        # over a possibly-mutated target; an apply could land changes twice).
        # CR2-1: the explicit owner acknowledgment re-enters the NORMAL flow.
        inspect_hint = (
            "compare the live payload content against the patch artifact"
            if getattr(entry, "authority_source", "") == "skill_payload"
            else "vcs_diff, compare against the patch artifact")
        return (
            f"⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: a durable apply-intent row "
            f"exists for run {rid} but no completed disposition — a previous "
            f"process may have applied this patch into {entry.target_root} before "
            "dying. The target MAY already carry the run's changes: inspect it "
            f"({inspect_hint}), then re-run "
            "integrate_delegated_patch with acknowledge_ambiguous=true to take "
            "the state over explicitly — that resolves the stale intent and runs "
            "the NORMAL disposition from scratch (an apply re-verifies baseline "
            "drift and honestly refuses a target that already moved; a reject "
            "releases the snapshot while the captured patch artifact is "
            "retained). Nothing was changed now; the execution snapshot and the "
            "captured patch are preserved."
        )
    return ""


def _unwritten_disposition_text(rid: str, target_root: str, disposition: str,
                                applied: bool, *, payload: bool = False) -> str:
    """The typed refusal for a completed operation whose row did not land.
    ``payload`` selects accurate apply wording (Sol P2-3): a payload apply lands
    LIVE in the non-Git payload — nothing staged, no index for vcs_diff."""
    if applied:
        landed, verify = (
            (f"applied LIVE into the non-Git payload {target_root}",
             "compare the live payload against the patch artifact") if payload
            else (f"applied and staged in {target_root}", "verify with vcs_diff"))
        return (
            f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch IS {landed}, "
            "but the durable disposition row could not be "
            "written, so nothing on disk records that this patch was handled. Do "
            "NOT call integrate_delegated_patch again for this run — a second "
            "apply would land the same changes twice. Fix the drive/event log, "
            f"then {verify} and record the outcome; the execution "
            "snapshot is deliberately preserved."
        )
    return (
        f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch was NOT applied "
        f"(disposition {disposition!r}), and the durable disposition row could not "
        "be written. Nothing in your tree changed and the execution snapshot is "
        "preserved. Fix the drive/event log; this process already holds the "
        "disposition in memory (a repeat here answers ALREADY_DISPOSED), so the "
        "run reads as undisposed again only after a restart — repeat it then."
    )


def _dispose_delegated(drive: Any, entry: Any, snapshot_key: str, reason: str,
                       disposition: str, cleanup: bool) -> tuple[bool, str]:
    """Record a delegated disposition durably; clean up ONLY if the row landed.
    Releasing snapshot/patch on an UNWRITTEN row loses the record that the patch
    was handled (a restart could apply it twice). Shared by the Git and payload
    branches (``delegate_integration``). Returns ``(recorded, note)``."""
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import remove_execution_snapshot

    recorded = custody.record_patch_disposed(
        drive, entry, disposition=disposition, reason=str(reason or ""))
    if not recorded:
        return False, ""
    note = ""
    if cleanup:
        try:
            removed = remove_execution_snapshot(snapshot_key)
        except Exception:
            removed = False
        note = "" if removed else " (snapshot cleanup deferred to the startup GC.)"
    return True, note


def _resolve_acknowledged_intent(drive: Any, entry: Any) -> None:
    """CR2-1: the owner explicitly took over the AMBIGUOUS crash state.
    Resolves the stale apply intent durably as owner-acknowledged; the caller
    re-runs the NORMAL disposition from scratch (apply re-proves drift, reject
    re-runs the ready-manifest guard). A failed row write only makes a
    post-restart replay ambiguous again — the fail-closed direction."""
    from ouroboros import delegate_custody as custody

    custody.record_patch_apply_resolved(drive, entry, reason="owner_acknowledged")


def _integrate_delegated_patch(
    ctx: ToolContext,
    run_id: str = "",
    decision: str = "apply",
    reason: str = "",
    acknowledge_ambiguous: bool = False,
) -> str:
    """The C1 explicit acceptance seam: apply or reject ONE delegated run's captured patch.

    A mutating delegated run executed in a PRIVATE execution snapshot; its diff was
    captured at terminal. NOTHING reaches the shared tree automatically — this tool is
    the only path in, it targets the run's recorded authority target (which must be
    this task's own active root), and under the repo git lock proves no touched path
    drifted from the run's baseline, applies the patch to the WORKING TREE (a plain
    `git apply`, deliberately NOT --3way — see `_locked_apply`: --3way implies index
    binding and refuses the normal dirty shared-tree state the baseline deliberately
    snapshots), stages the touched paths explicitly, and records the disposition
    durably. Only a recorded disposition releases the snapshot for cleanup; a
    conflict keeps the snapshot and the patch as resolution material.
    ``acknowledge_ambiguous`` (CR2-1) is the owner exit from the AMBIGUOUS
    crash state: it resolves the stale intent durably and re-runs this normal
    flow, whose own guards re-verify the tree. A no-op when nothing is pending.
    """
    from ouroboros import delegate_custody as custody, delegate_source_coverage

    rid = str(run_id or "").strip()
    if not rid:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): run_id is required."
    decision = str(decision or "apply").strip().lower()
    if decision not in {"apply", "reject"}:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): decision must be 'apply' or 'reject'."
    drive = custody.custody_root(ctx)
    status, entry = custody.lookup(drive, str(getattr(ctx, "task_id", "") or ""), rid)
    refusal = _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous)
    if refusal:
        return refusal
    if decision == "apply" and entry is not None:
        if refusal := delegate_source_coverage.source_apply_refusal(entry, rid):
            return refusal
    if entry.patch_apply_pending and acknowledge_ambiguous:
        _resolve_acknowledged_intent(drive, entry)
    snapshot_key = entry.snapshot_id or entry.run_id
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, snapshot_key)
    manifest_path = cap_dir / "workspace_patch.json"
    patch_path = cap_dir / "workspace.patch"
    capture_refusal = _capture_at_disposition(drive, entry, rid, manifest_path)
    if capture_refusal:
        return capture_refusal
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
    if entry.authority_source == "skill_payload" or str(manifest.get("capture_kind") or "") == "skill_payload":
        # The exact-payload branch (R1 item 3) lives in delegate_integration:
        # fresh semantic rebinding, whole-payload CAS, reserved-path whole-apply
        # refusal, index-free apply into the live NON-Git payload — no staging.
        from ouroboros.tools.delegate_integration import integrate_payload_patch

        return integrate_payload_patch(
            ctx, drive=drive, entry=entry, rid=rid, decision=decision,
            reason=reason, cap_dir=cap_dir, manifest=manifest, patch_path=patch_path)
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    touched += [str(p) for p in (manifest.get("untracked_included") or [])]

    def _dispose(disposition: str, cleanup: bool) -> tuple[bool, str]:
        return _dispose_delegated(drive, entry, snapshot_key, reason, disposition, cleanup)

    def _unwritten_disposition(disposition: str, applied: bool) -> str:
        return _unwritten_disposition_text(rid, str(entry.target_root), disposition, applied)

    capture_status = str(manifest.get("status") or "")
    if decision == "reject":
        # A reject RELEASES the snapshot (the child's only copy): ready-only.
        if capture_status not in _READY_CAPTURE_STATUSES:
            return _capture_failed_refusal(
                rid, capture_status, "a reject would release the snapshot over it")
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[],
            target=str(entry.target_root),
        )
        recorded, note = _dispose("rejected", cleanup=True)
        if not recorded:
            return _unwritten_disposition("rejected", applied=False)
        return (
            f"🚫 Rejected delegated run {rid}'s captured patch ({len(touched)} file(s) not "
            f"applied); its execution snapshot is released. Verdict: {verdict_path or '(unwritten)'}. "
            f"Reason: {reason or '(none)'}.{_format_patch_exclusions(manifest)}{note}"
        )

    if capture_status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        if capture_status == ARTIFACT_STATUS_READY_NO_CHANGES:
            recorded, note = _dispose("applied", cleanup=True)
            if not recorded:
                return _unwritten_disposition("applied", applied=False)
            return (
                f"OK: delegated run {rid} changed NOTHING in its execution snapshot; "
                f"there is no patch to apply and the snapshot is released."
                f"{_format_patch_exclusions(manifest)}{note}"
            )
        return (
            f"⚠️ INTEGRATE_DELEGATED_NO_CAPTURE: run {rid}'s capture status is "
            f"{capture_status or 'missing'!r} — no applicable patch. If the run just "
            "ended, delegate_wait it once more to capture; a failed capture keeps the "
            "snapshot for direct inspection."
        )
    if not patch_path.exists():
        return f"⚠️ INTEGRATE_PATCH_MISSING: captured patch not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if expected_digest:
        actual_digest = _sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for run {rid} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )

    # The target is the run's RECORDED authority target, and it must be THIS task's
    # own active root — the nanny integrates into its own tree, never across trees.
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    target = pathlib.Path(str(entry.target_root or "")).resolve(strict=False)
    if not str(entry.target_root or "").strip() or target != active_root:
        return (
            "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run's recorded authority target "
            f"({entry.target_root or '(none)'}) is not this task's active root ({active_root}). "
            "Refusing to apply across trees."
        )
    if not (target / ".git").exists():
        return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."

    patch_touched, parse_error = _patch_touched_paths(patch_path, target)
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse run {rid}'s captured patch "
            f"(git apply --numstat failed): {parse_error[:300]}"
        )
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
    runtime_mode = get_runtime_mode()
    # The protected-path policy is about the OUROBOROS body's own invariants, so it
    # is asked only when the target IS that body (or a self_worktree checkout of
    # it). A foreign project's `.github/workflows/ci.yml` or `build.sh` is that
    # project's file: gating it here would block the root-delegation lane (B5) with
    # advice about a runtime mode that does not govern that repository — the same
    # reason `_handle_external_workspace_integration` never applies this gate.
    protected = (
        protected_paths_in(sorted(patch_touched)) if _target_is_system_repo(ctx) else []
    )
    if protected:
        grant_ok = (not is_acting) or bool(getattr(constraint, "protected_paths_grant", False))
        if not (mode_allows_protected_write(runtime_mode) and grant_ok):
            _write_verdict(
                ctx, f"run_{rid}", outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[],
                protected=[p.path for p in protected], target=str(target),
            )
            return protected_write_block_message(
                path=protected[0].path,
                runtime_mode=runtime_mode,
                action=f"integrate delegated run {rid} patch touching",
            )

    ordered_touched = sorted(patch_touched)
    # CR1-3, owed-before-sent: the durable apply-intent row lands BEFORE the tree
    # can be mutated. Without it, a crash between `git apply` and the disposition
    # row replays as "never applied", and a later reject records a false rejection
    # over a modified, staged tree. An unlanded intent row refuses the mutation
    # outright (same doctrine as record_start_requested).
    if not custody.record_patch_apply_started(drive, entry, target_root=str(target)):
        return (
            f"⚠️ INTEGRATE_INTENT_UNWRITTEN: the durable apply-intent row for run "
            f"{rid} could not be written, so a crash mid-apply would leave the "
            "tree state unaccountable. Refusing to mutate; fix the drive/event "
            "log and retry. Nothing was changed."
        )
    outcome = _locked_apply(ctx, target, patch_path, ordered_touched, entry.baseline_sha)
    if outcome.get("lock_error"):
        custody.record_patch_apply_resolved(drive, entry, reason="lock_error")
        return (
            "⚠️ INTEGRATE_LOCK_TIMEOUT: could not acquire the repo git lock: "
            f"{outcome['lock_error']}."
        )
    proc = outcome["proc"]
    drifted = outcome["drifted"]
    drift_error = outcome["drift_error"]
    staging_failure = outcome["staging_failure"]
    reverted = outcome["reverted"]

    if proc is None:
        # Nothing was attempted (proven drift / unverifiable baseline): the tree
        # is unmutated, so the intent resolves and the retry lane stays open.
        custody.record_patch_apply_resolved(drive, entry, reason="baseline_drift")
        return _drift_refusal(
            ctx, rid=rid, reason=reason, touched=touched, manifest=manifest,
            protected=protected, target=target, drifted=drifted, drift_error=drift_error,
        )

    if staging_failure:
        if reverted:
            # Cleanly reversed: the tree is PROVABLY back to pre-apply, so the
            # intent resolves BEFORE the verdict write — `_write_verdict` can
            # raise (artifact-dir mkdir), and a stranded pending intent over a
            # non-mutated tree would wedge the run into AMBIGUOUS (CR2-3).
            custody.record_patch_apply_resolved(drive, entry, reason="apply_reverted")
        outcome = "applied_unstaged_reverted" if reverted else "applied_unstaged"
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome=outcome, reason=reason, files=touched,
            manifest=manifest, applied=not reverted, conflicts=[staging_failure[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        if reverted:
            return (
                f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s patch applied cleanly into "
                f"{target} but STAGING it failed ({staging_failure[:300]}), so the apply "
                "was reversed — your tree is back to its pre-apply state and NOTHING is "
                "left half-applied. The snapshot and the patch are preserved; fix the "
                f"index problem, then call this tool again. Verdict: {verdict_path or '(unwritten)'}."
                f"{_format_patch_exclusions(manifest)}"
            )
        recorded, _ = _dispose("applied", cleanup=False)
        tail = "" if recorded else (
            " ⚠️ the durable disposition row could ALSO not be written — record this "
            "outcome yourself before any further integration attempt."
        )
        return (
            f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s patch IS APPLIED in {target} "
            f"({len(touched)} file(s)) but could NOT be staged ({staging_failure[:300]}), "
            "and the apply could not be cleanly reversed. Do NOT retry this call — the "
            "changes are already in your working tree and a second apply would double "
            "them. Inspect with vcs_diff, stage what you accept yourself, and note that "
            "the run is recorded as applied. Its execution snapshot is preserved for "
            f"comparison. Verdict: {verdict_path or '(unwritten)'}.{_format_patch_exclusions(manifest)}{tail}"
        )

    if proc.returncode != 0:
        # `git apply` is atomic (all-or-nothing without --reject): a non-zero exit
        # means the tree is unmutated, so the intent resolves and retry stays open.
        custody.record_patch_apply_resolved(drive, entry, reason="apply_failed")
        stderr = (proc.stderr or proc.stdout or "").strip()
        conflicts = [ln.strip() for ln in stderr.splitlines()
                     if "conflict" in ln.lower() or "patch failed" in ln.lower()]
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="conflict", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=conflicts or [stderr[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        return (
            f"⚠️ INTEGRATE_CONFLICT: applying run {rid}'s patch into {target} did not "
            f"apply cleanly. git said: {stderr[:600]}\n"
            "YOU own this conflict: the execution snapshot and the captured patch are "
            "preserved until you resolve it. Reconcile your tree with the snapshot's "
            "changes (read the patch artifact, or diff against the execution root "
            "directly), retry the apply, or "
            "integrate_delegated_patch(decision='reject') to discard. "
            f"Verdict: {verdict_path or '(unwritten)'}."
        )

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_delegated_patch",
        )
    except Exception:
        pass
    verdict_path = _write_verdict(
        ctx, f"run_{rid}", outcome="applied", reason=reason, files=touched,
        manifest=manifest, applied=True, conflicts=[],
        protected=[p.path for p in protected], target=str(target),
    )
    recorded, note = _dispose("applied", cleanup=True)
    if not recorded:
        return _unwritten_disposition("applied", applied=True)
    diffstat = str(manifest.get("diffstat") or "").strip()
    prot_note = ""
    if protected:
        prot_note = f" Includes {len(protected)} protected path(s) (allowed: runtime_mode={runtime_mode})."
    return (
        f"✅ Integrated delegated run {rid}'s patch into {target} ({len(touched)} file(s), staged).{prot_note}\n"
        f"{diffstat}{_format_patch_exclusions(manifest)}\n"
        f"Verdict: {verdict_path or '(unwritten)'}. Its execution snapshot is released.\n"
        "Changes are staged but NOT committed — review them yourself; you are the sole committer."
        f"{note}"
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
            f"- diffstat: {diffstat or '(none)'}{_format_patch_exclusions(manifest)}\n"
            + (f"- child summary: {result_summary}\n" if result_summary else "")
            + (f"\n```diff\n{body}\n```\n" if body else "- (no patch body; nothing to apply)\n")
        )
    parts.append(
        "\nPick the best candidate and apply it with integrate_subagent_patch(task_id=...), "
        "or synthesize across candidates yourself (you are the sole committer). Comparison is read-only."
    )
    return "\n".join(parts)


def _format_patch_exclusions(manifest) -> str:
    """One disclosure line for files the capture dropped per policy, or ''.

    The per-file exclusions (#447 F5) live in the workspace_patch manifest,
    which no parent-facing surface used to render — a dropped deliverable
    hid behind an affirmative "Integrated N file(s)" success line."""
    entries = []
    for key in ("sensitive_blocked", "untracked_excluded", "tracked_excluded"):
        for item in manifest.get(key) or []:
            if isinstance(item, dict):
                path, reason = str(item.get("path") or ""), str(item.get("reason") or "")
            else:
                path, reason = str(item or ""), ""
            if path:
                entries.append(f"{path} ({reason})" if reason else path)
    if not entries:
        return ""
    shown = "; ".join(entries[:8])
    more = (
        f" and {len(entries) - 8} more (full list with reasons: the child's "
        "workspace_patch.json artifact)"
        if len(entries) > 8 else ""
    )
    return (
        f"\n⚠️ {len(entries)} file(s) EXCLUDED from this patch by capture policy: "
        f"{shown}{more}. Excluded content is NOT in this patch: if one is a real "
        "deliverable, recover it from the child workspace/snapshot while that "
        "still exists, or have it re-produced."
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
        ToolEntry(
            "integrate_delegated_patch",
            {
                "name": "integrate_delegated_patch",
                "description": (
                    "EXPLICITLY apply or reject the captured patch of ONE of your own delegated "
                    "runs (delegate_start). A mutating delegated run edits a PRIVATE execution "
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
                    "not a verified result."
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
