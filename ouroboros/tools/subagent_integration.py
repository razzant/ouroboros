"""``integrate_subagent_patch``: the parent's manifest-first integration tool.

A mutative (acting) subagent returns its changes as a ``workspace.patch`` artifact
(produced by headless finalization, a git diff against the child's base commit).
The parent decides what to do with it — accept one (best-of-N), synthesize several,
or reject. This tool applies isolated self_worktree patches to the parent's repo,
or verifies native external_workspace changes at the child’s recorded shared target
under current read authority, even when the parent works elsewhere. Nothing is
transferred into the parent’s folder.
The parent stays the sole committer: applying stages changes but never
commits; the parent reviews and runs ``commit_reviewed`` itself.

Only the immediate parent integrates a child. A bound copy returns to its
recorded source under the parent's current write rights; legacy patches target
the active repo. Descendant patches still bubble up one level at a time.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any, Dict, Iterator, List, Tuple, Union

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import completed_local_read
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.task_results import load_task_result
from ouroboros.workspace_file_outputs import file_output_changes, prepare_file_outputs, verify_file_outputs
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


def _granted_root_bindings(ctx: ToolContext, ceiling: Any, binding: Any) -> Iterator[Any]:
    """Other ceiling-granted roots whose ordinary binding reaches this same physical file.

    Overlapping roots (Deliverables inside user_files) give one file several ordinary
    labels; the deepest containing one must not discard a grant on another. Each
    candidate passes the ordinary resolver and its confinement; nothing is aliased.
    """
    from ouroboros.tool_access import build_resolved_resource_binding

    physical = pathlib.Path(binding.target_path).resolve(strict=False)
    for root in dict.fromkeys(g.root for g in ceiling.resource_grants if "read" in g.operations):
        if root == binding.root:
            continue
        try:
            other = build_resolved_resource_binding(ctx, root=root, operation="read", path=str(physical))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if pathlib.Path(other.target_path).resolve(strict=False) == physical:
            yield other


def _shared_read_refusal(ctx: ToolContext, target: pathlib.Path, operation: str = "") -> str:
    """Use current read authority; selecting a child's folder adds no grant."""
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path
    from ouroboros.presence_authority import presence_ceiling_from_context, presence_ceiling_allows_binding
    from ouroboros.protected_artifacts import _artifact_records, _operation_denied, block_reason_for_path

    try:
        root = _root_containing_absolute_path(ctx, "read_file", str(target))
        binding = build_resolved_resource_binding(
            ctx, root=root or "user_files", operation="read", path=str(target))
        ceiling = presence_ceiling_from_context(ctx)
        if ceiling is not None and not presence_ceiling_allows_binding(ceiling, binding) and not any(
                presence_ceiling_allows_binding(ceiling, other)
                for other in _granted_root_bindings(ctx, ceiling, binding)):
            return "Presence resource ceiling does not allow reading this child target"
        if operation:
            # Only relative policy needs the caller's active root (including a
            # project room). B's read binding must never become the policy base.
            relative_policy = any(
                _operation_denied(record, operation)
                and any(str(path).strip() and not pathlib.Path(str(path).strip()).expanduser().is_absolute()
                        for path in record.get("paths") or [])
                for record in _artifact_records(ctx))
            policy_binding = (build_resolved_resource_binding(
                ctx, root="active_workspace", operation="read", path=".") if relative_policy else None)
            if refusal := block_reason_for_path(ctx, target, operation, policy_binding):
                return refusal
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return str(exc)
    return ""


def _shared_target(child_result: Dict[str, Any], manifest: Dict[str, Any], requested: str) -> pathlib.Path:
    """Canonical assignment selects B; present capture/assignment fields must agree.

    Historical captures may omit additive root/base fields. Neither an explicit
    target nor a manifest alone selects a destination. Shared live HEAD may move.
    """
    from ouroboros.workspace_copies import same_directory

    root = _child_write_root(child_result)
    if not root:
        raise ValueError("child result did not record write_root/workspace_root")
    target = pathlib.Path(root).expanduser().resolve(strict=False)
    constraint = child_result.get("task_constraint") if isinstance(child_result.get("task_constraint"), dict) else {}
    metadata = child_result.get("metadata") if isinstance(child_result.get("metadata"), dict) else {}
    for value in (constraint.get("write_root"), child_result.get("workspace_root"),
                  metadata.get("workspace_root"), child_result.get("write_root"),
                  manifest.get("workspace_root"), requested):
        if value and not same_directory(value, target):
            raise ValueError("recorded child target, capture workspace_root or explicit target_root disagree")
    base, captured = constraint.get("base_sha"), manifest.get("base_head")
    if base and captured and base != captured:
        raise ValueError("capture base does not match the child's admitted base")
    return target


def _verify_shared_external_workspace(
    target: pathlib.Path,
    patch_path: pathlib.Path,
    touched: List[str],
    file_rows: List[Dict[str, Any]] = (),
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
    try:
        if not verify_file_outputs(file_rows, target):
            return False, [], "registered file outputs do not match the shared workspace"
    except (OSError, ValueError) as exc:
        return False, [], str(exc)
    if not patch_path.is_file() or not patch_path.stat().st_size:
        return (True, [], "") if file_rows else (False, [], "workspace patch and file outputs are absent")
    from ouroboros.repo_diff_capture import _GIT_RETARGET_ENV
    # Capture diffed B under the ambient Git configuration (CRLF conversion, filters);
    # the read-only reverse check must convert the same way, so it strips only the
    # variables that would redirect Git to another repository or index.
    proc = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=str(target),
        capture_output=True,
        text=True,
        env={**{k: v for k, v in os.environ.items() if k not in _GIT_RETARGET_ENV},
             "GIT_CEILING_DIRECTORIES": str(resolved_target.parent)},
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, [], detail[:600] or "reverse patch check failed"
    return True, [], ""


def _capped_self_repo_refusal(ctx: Any, child_task_id: str) -> str:
    """A tree under the light per-task cap (a consciousness Act/Observe tree) may not land a
    patch on the Ouroboros repository — protected paths or not, in every install mode. The
    empty string when the task is not capped."""
    from ouroboros.consciousness_authority import task_mode_capped_light

    if not task_mode_capped_light(getattr(ctx, "task_metadata", None)):
        return ""
    return (
        f"⚠️ INTEGRATE_CAPPED_TREE: child {child_task_id} produced a self_worktree patch (against "
        "the Ouroboros system repo), but this task's tree runs under a light cap (a consciousness "
        "Act/Observe tree): it may not land a patch on the Ouroboros repository, protected paths "
        "or not, in any runtime mode."
    )


def _integration_runtime_mode(ctx: Any) -> str:
    """The mode the protected-path gate of an integration reads: the stricter of the install
    mode and the task's own cap (a consciousness Act/Observe tree is light), so a capped task
    cannot land a system-repo patch the install mode alone would allow."""
    from ouroboros.consciousness_authority import effective_runtime_mode

    return effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None))


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
    the registry's own write gate: absent workspace means the live body; an
    isolated copy retains its admitted source identity. Legacy copies are
    interpreted as the body, as their original contract required.
    """
    from ouroboros.workspace_copies import copy_binding, source_is_system_repo
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if ((getattr(constraint, "surface", "") == "self_worktree"
         or getattr(ctx, "workspace_mode", "") == "self_worktree") and not copy_binding(ctx)):
        return True  # Legacy isolated records were always own-body copies.
    try:
        return source_is_system_repo(ctx.active_repo_dir(), system_repo_dir_for(ctx))
    except (AttributeError, OSError, ValueError, TypeError):
        return True


def _is_host_minted_projects_tree(path: pathlib.Path) -> bool:
    """True when ``path`` is a host-minted genesis/coop tree — i.e. inside the
    durable subagent-projects root. Owner-attached folders never live there, so this
    is the structural boundary for the coop no-op (and the checkpoint-commit).
    It asks the SAME containment predicate the orphan apply gate asks
    (``delegate_shared.orphan_apply_target_ok``); here the projects root is both
    the boundary and the containment root, so the two can never drift apart."""
    try:
        from ouroboros.config import get_subagent_projects_root
        from ouroboros.delegate_shared import orphan_apply_target_ok

        return orphan_apply_target_ok(path, get_subagent_projects_root())
    except Exception:
        return False


def _verify_directory_direct_result(
    ctx: ToolContext, child_task_id: str, reason: str, target: pathlib.Path,
    manifest: Dict[str, Any], artifact_dir: pathlib.Path,
) -> str:
    """Verify only registered postimages; direct effects are never replayed or rolled back."""
    from ouroboros.artifacts import stream_artifact_file

    verified: set[str] = set()
    try:
        if manifest.get("status") != "ready" or manifest.get("apply_state") != "already_applied":
            raise ValueError("direct folder capture is not ready")
        if pathlib.Path(str(manifest.get("workspace_root") or "")).resolve(strict=False) != target:
            raise ValueError("direct folder capture root does not match its recorded child target")
        outputs = manifest.get("registered_outputs")
        if not isinstance(outputs, list):
            raise ValueError("registered output records are unavailable")
        # A directory registers a member ledger (`<kind>_manifest`) plus its `<kind>`
        # zip package; every other record names one file. Each takes its own check.
        directories = {(str(item.get("kind") or "").removesuffix("_manifest"), str(item.get("source_path") or ""))
                       for item in outputs if str(item.get("kind") or "").endswith("_manifest")}
        for item in outputs:
            source = pathlib.Path(item["source_path"]).resolve(strict=False)
            source.relative_to(target)
            artifact = pathlib.Path(item["path"]).resolve(strict=False)
            artifact.relative_to(artifact_dir.resolve(strict=False))
            if not item.get("sha256") or not isinstance(item.get("size"), int):
                raise ValueError("registered output is missing its captured identity")
            if refusal := _shared_read_refusal(ctx, source, "read_bytes"):
                raise ValueError(refusal)
            stream_artifact_file(artifact, expected=item)
            kind = str(item.get("kind") or "")
            directory = kind.endswith("_manifest") or (kind, str(item.get("source_path") or "")) in directories
            if directory and not source.is_dir():
                raise ValueError(f"registered directory output is no longer a directory: {source}")
            if not directory and source.is_dir():
                raise ValueError(f"registered file output is now a directory: {source}")
            if kind.endswith("_manifest"):
                ledger = json.loads(artifact.read_text(encoding="utf-8"))
                if pathlib.Path(str(ledger.get("source_path") or "")).resolve(strict=False) != source:
                    raise ValueError("directory output ledger source does not match registration")
                for member in ledger["files"]:
                    path = (source / member["path"]).resolve(strict=False)
                    path.relative_to(source)
                    if not member.get("sha256") or not isinstance(member.get("size"), int):
                        raise ValueError("directory member is missing its captured identity")
                    for operation in ("read_bytes", "hash"):
                        if refusal := _shared_read_refusal(ctx, path, operation):
                            raise ValueError(refusal)
                    stream_artifact_file(path, expected=member)
                    verified.add(path.relative_to(target).as_posix())
            elif not directory:
                if refusal := _shared_read_refusal(ctx, source, "hash"):
                    raise ValueError(refusal)
                stream_artifact_file(source, expected=item)
                verified.add(source.relative_to(target).as_posix())
        outcome = "verified_registered_outputs" if verified else "direct_result_observed"
        detail = (f"Verified {len(verified)} registered file postimage(s) in {target}. " if verified else
                  f"Recorded the parent's acceptance of the direct child result in {target}; no file postimages were verified. ")
        detail += "Other shell, GUI or external effects and the complete changed-file set remain unknown. No effects were re-applied, transferred to the parent, or rolled back."
        conflicts = []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        outcome, detail = "direct_output_mismatch", f"Registered output verification failed: {exc}"
        conflicts = [str(exc)]
    verdict = _write_verdict(
        ctx, child_task_id, outcome=outcome, reason=f"{reason + '. ' if reason else ''}{detail}",
        files=sorted(verified), manifest=manifest, applied=False, conflicts=conflicts, protected=[], target=str(target),
    )
    if conflicts:
        return f"⚠️ INTEGRATE_DIRECTORY_OUTPUT_MISMATCH: {detail}. Verdict: {verdict or '(unwritten)'}."
    warning = _record_integration_disposition(ctx, child_task_id, "integrated", reason, detail)
    return f"OK: {detail} Verdict: {verdict or '(unwritten)'}.{warning}"


def _handle_external_workspace_integration(
    ctx: ToolContext,
    *,
    child_task_id: str,
    reason: str,
    requested_target: str,
    patch_path: pathlib.Path,
    manifest: Dict[str, Any],
    child_result: Dict[str, Any],
    touched: List[str],
    file_rows: List[Dict[str, Any]] = (),
) -> str:
    def refused(code: str, outcome: str, detail: str, target: Any = "",
                files: List[str] = touched, conflicts: List[str] = ()) -> str:
        """An early refusal keeps its audit verdict and custody row; no disposition."""
        verdict = _write_verdict(
            ctx, child_task_id, outcome=outcome, reason=reason or detail, files=files, manifest=manifest,
            applied=False, conflicts=list(conflicts) or [detail], protected=[], target=str(target or ""))
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
        text = (f"⚠️ {code}: {detail}. Verdict: {verdict or '(unwritten)'}. "
                "Captured result retained; nothing was transferred or marked integrated.")
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="INTEGRATION_BLOCKED", text=text, meta={"identifier": code}))

    def forbidden(refusal: str) -> str:
        return (f"{refusal.rstrip('. ')}. Verification reads the child's folder under your current read, Presence "
                "and task-policy authority; the assignment grants none. Inspect or reject the result")

    assigned = _child_write_root(child_result)
    try:
        target = _shared_target(child_result, manifest, requested_target)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if not assigned:  # Nothing names a target; never substitute the parent's folder.
            return refused("INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISSING", "shared_workspace_missing_target",
                           f"child {child_task_id} did not record its assigned write_root/workspace_root")
        return refused("INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH", "shared_workspace_target_mismatch",
                       f"{exc}; verification runs only at the recorded assignment {assigned}", target=assigned)
    if refusal := _shared_read_refusal(ctx, target):
        return refused("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused", forbidden(refusal), target)

    if manifest.get("capture_kind") == "directory_direct":
        return _verify_directory_direct_result(
            ctx, child_task_id, reason, target, manifest, patch_path.parent,
        )
    # Parse all captured paths without an ancestor repo's subdirectory prefix. Git
    # ignores a ceiling equal to its cwd and never searches the ceiling itself, so the
    # artifact folder's parent stops discovery right above the artifact folder.
    from ouroboros.subagent_worktrees import isolated_git_env
    parse_env = {**isolated_git_env(), "GIT_CEILING_DIRECTORIES": str(patch_path.parent.resolve().parent)}
    patch_touched, parse_error = (_patch_touched_paths(patch_path, patch_path.parent, env=parse_env)
                                  if patch_path.is_file() and patch_path.stat().st_size else (set(), ""))
    if parse_error:
        return refused("INTEGRATE_PATCH_UNREADABLE", "shared_workspace_patch_unreadable",
                       f"cannot parse {child_task_id} workspace.patch: {parse_error[:300]}", target)
    authoritative_touched = sorted(patch_touched | {row["path"] for row in file_rows} or set(touched))
    hashed = {row["path"] for row in file_rows}
    paths = {rel: (target / rel).resolve(strict=False) for rel in authoritative_touched}
    if escaped := [rel for rel, path in paths.items() if not path.is_relative_to(target)]:
        return refused("INTEGRATE_EXTERNAL_WORKSPACE_MISSING", "shared_workspace_missing",
                       f"child {child_task_id} result names {len(escaped)} path(s) outside {target}; "
                       f"first 20: {escaped[:20]}; omitted: {max(0, len(escaped) - 20)}",
                       target, authoritative_touched, escaped)
    for rel, path in paths.items():
        for operation in (("read_bytes", "hash") if rel in hashed else ("read_bytes",)):
            if refusal := _shared_read_refusal(ctx, path, operation):
                return refused("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused",
                               forbidden(refusal), target, authoritative_touched)
    verified, missing, detail = _verify_shared_external_workspace(
        target, patch_path, authoritative_touched, file_rows)
    coop = _is_host_minted_projects_tree(target)
    outcome = ("coop_already_in_tree" if coop else "verified_shared_workspace") if verified else (
        "shared_workspace_missing" if missing else "shared_workspace_mismatch")
    verdict = _write_verdict(
        ctx, child_task_id, outcome=outcome, reason=reason, files=authoritative_touched,
        manifest=manifest, applied=False, conflicts=missing or ([detail] if detail else []),
        protected=[], target=str(target))
    if verified:
        warning = _record_integration_disposition(
            ctx, child_task_id, "integrated", reason,
            "verified child result at its recorded shared target; no transfer to parent")
        prefix = ("OK: cooperative no-op — work is ALREADY in the shared coop tree. " if coop else
                  f"✅ Verified external_workspace child {child_task_id}: ")
        checkpoint = " The host attempts a best-effort checkpoint commit of this tree when the root task finalizes." if coop else ""
        return (f"{prefix}{len(authoritative_touched)} file(s) verified in {target}. "
                f"No patch was re-applied or transferred to the parent's folder.{checkpoint} "
                f"Verdict: {verdict or '(unwritten)'}.{_format_patch_exclusions(manifest)}{warning}")
    if missing:
        return (f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISSING: child {child_task_id} result does not verify in {target}: "
                f"{missing}. Verdict: {verdict or '(unwritten)'}. Captured result retained.")
    return (f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH: child {child_task_id} result does not verify in {target}: "
            f"{detail}. Verdict: {verdict or '(unwritten)'}. Captured result retained.")


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
        direct_note = (" Direct effects remain in the folder; rejecting this result does not undo them."
                       if manifest.get("capture_kind") == "directory_direct" else "")
        return (
            f"🚫 Rejected subagent patch from {child_task_id} ({len(touched)} file(s) not applied).{direct_note} "
            f"Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}."
            f"{_format_patch_exclusions(manifest)}{disposition_warning}"
        )

    if manifest.get("capture_kind") == "directory_direct":
        if child_surface != "external_workspace":
            return "⚠️ INTEGRATE_DIRECTORY_SURFACE_MISMATCH: direct folder results require external_workspace."
        return _handle_external_workspace_integration(
            ctx, child_task_id=child_task_id, reason=reason, requested_target=str(target_root or "").strip(),
            patch_path=patch_path, manifest=manifest, child_result=child_result, touched=touched,
        )

    status = str(manifest.get("status") or "")
    if status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        return (
            f"⚠️ INTEGRATE_NO_CHANGES: child {child_task_id} workspace patch status={status!r}; "
            "nothing to apply."
            f"{_format_patch_exclusions(manifest)}"
        )
    try:
        file_rows = file_output_changes(manifest, patch_path.parent)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        return f"⚠️ INTEGRATE_FILE_OUTPUTS_UNAVAILABLE: {exc}"
    has_patch = patch_path.is_file() and patch_path.stat().st_size > 0
    if not has_patch and (not file_rows or manifest.get("patch_size")):
        return f"⚠️ INTEGRATE_PATCH_MISSING: workspace.patch for {child_task_id} not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if has_patch and expected_digest:
        actual_digest = _sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for {child_task_id} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )
    touched = sorted(set(touched) | {row["path"] for row in file_rows})

    if child_surface == "external_workspace":
        return _handle_external_workspace_integration(
            ctx,
            child_task_id=child_task_id,
            reason=reason,
            requested_target=str(target_root or "").strip(),
            patch_path=patch_path,
            manifest=manifest,
            child_result=child_result,
            touched=touched, file_rows=file_rows,
        )


    from ouroboros.workspace_copies import copy_binding, same_directory, source_is_system_repo, copy_apply_refusal
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    child_copy = copy_binding(child_result) or child_result.get("workspace_copy") or {}
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    requested_target = str(target_root or "").strip()
    target = pathlib.Path(child_copy["source_root"]).resolve() if child_copy.get("source_root") else active_root
    if requested_target and not same_directory(requested_target, target):
        return "⚠️ INTEGRATE_TARGET_FORBIDDEN: target_root must name this copy's recorded source; legacy patches target the active root."
    if not (target / ".git").exists():
        return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."


    if child_surface == "self_worktree":
        if child_copy:
            if (not same_directory(child_copy.get("source_root"), target)
                    or child_copy.get("baseline_sha") != manifest.get("base_head")
                    or child_copy != manifest.get("workspace_copy")):
                return "⚠️ INTEGRATE_COPY_BINDING_MISMATCH: the recorded copy source, baseline or capture does not match this parent."
        elif ctx.is_workspace_mode() and str(getattr(ctx, "workspace_mode", "")) != "self_worktree":
            # Legacy records always meant an Ouroboros-body copy. Missing
            # provenance must never turn one into a foreign-project patch.
            return "⚠️ INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE: legacy system-repo patch cannot be applied to an external workspace."
        if source_is_system_repo(target, system_repo_dir_for(ctx)) or (not child_copy and _target_is_system_repo(ctx)):
            capped = _capped_self_repo_refusal(ctx, child_task_id)
            if capped:
                return capped

    runtime_mode = _integration_runtime_mode(ctx)
    # Derive the changed-path set from the PATCH ITSELF (not the child-controlled
    # manifest) for the protected-path gate: a child must not be able to hide a
    # protected edit by omitting it from the manifest (sha256 verifies bytes only).
    patch_touched, parse_error = _patch_touched_paths(patch_path, target) if has_patch else (set(), "")
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse {child_task_id} workspace.patch for the "
            f"protected-path check (git apply --numstat failed): {parse_error[:300]}"
        )
    touched = sorted(patch_touched | {row["path"] for row in file_rows})
    target_is_body = source_is_system_repo(target, system_repo_dir_for(ctx)) if child_copy else _target_is_system_repo(ctx)
    protected = protected_paths_in(touched) if target_is_body else []
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

    if child_copy:
        if refusal := copy_apply_refusal(ctx, target, touched):
            return f"⚠️ INTEGRATE_TARGET_FORBIDDEN: {refusal}"
        try:
            outcome = _locked_apply(
                ctx, target, patch_path, touched, child_copy["baseline_sha"],
                file_changes=file_rows, file_baseline=child_copy.get("file_baseline", {}),
                three_way=bool(child_copy.get("source_index_clean") and child_copy.get("target_head") == child_copy["baseline_sha"]),
                admission_check=lambda: copy_apply_refusal(ctx, target, touched))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            return f"⚠️ INTEGRATE_APPLY_UNKNOWN: {type(exc).__name__}: {exc}. Inspect the target before another apply; the captured result is retained."
        detail = outcome.get("admission_refusal") or outcome.get("lock_error") or outcome.get("drift_error")
        if outcome.get("drifted"):
            detail = "source files changed since copy: " + ", ".join(outcome["drifted"])
        proc = outcome["proc"]
        apply_attempted = proc is not None
        partial_applied = bool(outcome["staging_failure"] and not outcome["reverted"])
        if detail or outcome["staging_failure"] or proc is None:
            proc = subprocess.CompletedProcess([], 1, "", detail or outcome["staging_failure"] or "copy apply was not attempted")
    else:
        # Legacy --index apply shares the reviewed-commit Git lock.
        from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

        try:
            _git_lock = _acquire_git_lock(ctx)
        except Exception as exc:
            return f"⚠️ INTEGRATE_LOCK_TIMEOUT: could not acquire the repo git lock: {type(exc).__name__}: {exc}."
        partial_applied = False
        apply_attempted = False
        try:
            # Match --index semantics for file results too: never replace a parent's
            # staged preimage merely because its working copy matches the child base.
            index_tree = subprocess.run(
                ["git", "write-tree"], cwd=str(target), capture_output=True, text=True, check=True,
            ).stdout.strip() if file_rows else ""
            with prepare_file_outputs(file_rows, target, baseline_sha=index_tree) as prepared:
                apply_attempted = has_patch
                proc = (subprocess.run(
                    ["git", "apply", "--3way", "--index", str(patch_path)],
                    cwd=str(target), capture_output=True, text=True,
                ) if has_patch else subprocess.CompletedProcess([], 0, "", ""))
                if proc.returncode == 0 and file_rows:
                    try:
                        apply_attempted = True
                        prepared.apply()
                        if not prepared.verify_applied():
                            raise OSError("file outputs changed before staging")
                        paths = _stageable_paths(target, prepared.paths)
                        if paths:
                            stage = subprocess.run(
                                ["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"],
                                cwd=str(target), capture_output=True,
                                input=b"\0".join(p.encode("utf-8", errors="surrogateescape") for p in paths) + b"\0",
                            )
                            if stage.returncode:
                                raise OSError((stage.stderr or stage.stdout).decode("utf-8", errors="replace"))
                    except Exception as exc:
                        partial_applied = has_patch
                        try:
                            prepared.rollback()
                            detail = "file output writes reverted; inspect any applied text patch before retrying"
                        except Exception as rollback_exc:
                            partial_applied = True
                            detail = f"file output rollback incomplete: {rollback_exc}"
                        proc = subprocess.CompletedProcess([], 1, "", f"{exc}; {detail}")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            proc = subprocess.CompletedProcess([], 1, "", str(exc))
        finally:
            _release_git_lock(_git_lock)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        conflicts = [ln.strip() for ln in stderr.splitlines() if "conflict" in ln.lower() or "patch failed" in ln.lower()]
        _write_verdict(
            ctx, child_task_id, outcome="partially_applied" if partial_applied else "conflict", reason=reason, files=touched,
            manifest=manifest, applied=partial_applied, conflicts=conflicts or [stderr[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        detail = ("Integration did not finish cleanly." if apply_attempted else
                  "Result preparation failed. No file or patch apply was attempted.")
        next_step = ("Inspect with vcs_diff and resolve, or run vcs_restore to abort, then retry or pick another child."
                     if apply_attempted else "The copy and patch are retained. Reconcile changed source files with this copy, or reject the result.")
        return (f"⚠️ INTEGRATE_CONFLICT: {child_task_id} into {target}: {detail} "
                f"Details: {stderr[:600]}\n{next_step}")

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_subagent_patch",
            mutating_task_id=str(getattr(ctx, "task_id", "") or ""),
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


@completed_local_read
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
            + (f"- direct result: {manifest['note']}\n" if manifest.get("capture_kind") == "directory_direct" else "")
            + (f"- file results: {len(manifest.get('file_output_changes') or [])} captured change(s)\n"
               if manifest.get("file_output_changes") else "")
            + (f"\n```diff\n{body}\n```\n" if body else
               "- No inline patch body; inspect the recorded result and file artifacts.\n")
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
                    "a source-bound apply/stage into its recorded source under your current write authority (clean copies support 3-way synthesis); native external_workspace "
                    "verifies files already in the shared tree WITHOUT reapplying. Genesis is a standalone "
                    "directory, not a repo patch. This never commits; self-modification still requires your "
                    "commit_reviewed. For best-of-N pick a child "
                    "and integrate it, or integrate several to synthesize. Own-body protected-path changes require "
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
                        "target_root": {"type": "string", "description": "Optional target assertion: external_workspace verifies at the child’s recorded folder under your current read authority, without transfer; an explicit target must equal that folder. Isolated copies return to their recorded source under current write authority; legacy patches target your active root."},
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
                    "EXPLICITLY apply or reject the captured result of ONE terminal delegated run. "
                    "Its starter or host-confirmed retry successor may decide it; terminal-owner "
                    "orphan recovery remains available to top-level tasks. Current target authority still applies. "
                    "Git runs edit a private execution snapshot; apply verifies its complete result "
                    "against the recorded baseline under the repository lock, applies and stages "
                    "changed files into your active root, and never commits. Skill-payload runs "
                    "apply LIVE apply into the non-Git payload with content-hash CAS (nothing is staged into your active root) and make its review "
                    "stale before reuse. Ordinary-folder copies use the engine's complete file "
                    "manifest and per-file baseline checks; paths may select files and remaining "
                    "results stay retained. Direct ordinary-folder work has already changed the "
                    "source: apply acknowledges those effects, reject cannot undo them, and no full "
                    "rollback is promised. Unapplied copies can be explicitly rejected; conflicts "
                    "retain their snapshot and results. Read the captured result before deciding. "
                    "Undisposed snapshots or directory-copy results remain custody debt; direct runs "
                    "do not create that debt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The delegated run whose captured patch to integrate (from delegate_start)."},
                        "decision": {"type": "string", "enum": ["apply", "reject"], "default": "apply", "description": "apply = integrate the captured result: Git changes are STAGED into your active root; skill changes are applied LIVE into the non-Git payload with content-hash CAS; directory copies use engine file delivery. reject = explicit discard of unapplied results, never an undo of direct effects."},
                        "paths": {"type": "array", "items": {"type": "string"}, "description": "For an engine directory copy, optionally select a nonempty list of captured file paths to apply; omit for the complete result. Remaining changes stay retained until applied or explicitly rejected. Git and skill captures are whole-result operations: omit paths or pass [] for the complete capture; nonempty selections are unsupported there."},
                        "reason": {"type": "string", "description": "Optional rationale recorded in the verdict and the durable disposition row."},
                        "acknowledge_ambiguous": {"type": "boolean", "default": False, "description": "Set true ONLY after inspecting an INTEGRATE_DELEGATED_APPLY_AMBIGUOUS state (a crashed apply left a durable unresolved intent): resolves that stale intent and re-runs the normal disposition guards, which re-verify the tree. A no-op when no ambiguity is pending."},
                    },
                    "required": ["run_id"],
                },
            },
            lambda ctx, run_id="", decision="apply", reason="", acknowledge_ambiguous=False, paths=None: _integrate_delegated_patch(
                ctx, run_id, decision, reason, acknowledge_ambiguous=bool(acknowledge_ambiguous), paths=paths),
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
