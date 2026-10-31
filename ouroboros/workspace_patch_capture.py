"""Workspace patch capture: the patch artifact, its manifest, and its git plumbing.

Owns the streamed `workspace.patch` and `workspace_patch.json` pair — patch
baseline resolution (including the unborn-HEAD empty-tree case and the acting
subagent `base_sha` binding), the bounded git process helpers the capture runs
on, the declared-scratch and untracked eligibility filtering (git's binary
verdict for a whole inventory in one process), the moved-HEAD
tripwire for a private self worktree, and the empty manifest a failed
finalization falls back to. The static eligibility rules live in
``workspace_patch_rules``; the task-drive, child-result and artifact
finalization owners stay with ``headless``.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import stat
import subprocess
import tempfile
import threading
from hashlib import sha256
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

from ouroboros.artifact_storage import compress_in_place
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.git_observation import numstat_paths
from ouroboros.headless_status import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.workspace_patch_rules import (
    NAMED_EVIDENCE_EXCLUSION,
    _PATCH_EXCLUDE_RULES_VERSION,
    _PATCH_MAX_UNTRACKED_FILE_BYTES,
    _incidental_lockfile_excludes,
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
_PATCH_FILE_REFERENCE_BYTES = 50 * 1024 * 1024


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

    from ouroboros.artifacts import stream_artifact_file

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
    file_output_paths: List[str] = []
    acting_constraint = _acting_constraint_from_task(task)
    task_base_sha = str(acting_constraint.base_sha or "").strip() if acting_constraint else ""
    preflight_head = _preflight_head_from_task(task)
    if not task_base_sha and not preflight_head and (acting_constraint or _preflight_head_present(task)):
        manifest = _empty_patch_manifest(root, status=ARTIFACT_STATUS_FAILED, errors=[{
            "type": "workspace_base_unknown",
            "message": "The required original Git basis was not recorded; the workspace is retained without a patch.",
        }])
        manifest.update(base_provenance="unknown", note=manifest["errors"][0]["message"])
        patch_path.unlink(missing_ok=True)
        atomic_write_json(manifest_path, manifest, trailing_newline=True)
        return [{"kind": "workspace_patch_manifest", "name": manifest_path.name,
                 "path": str(manifest_path), **stream_artifact_file(manifest_path),
                 "workspace_root": str(root)}], manifest
    base_ref, base_head, base_is_empty_tree = _workspace_patch_base(
        root,
        errors,
        expected_base_sha=task_base_sha or preflight_head,
    )
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    workspace_copy = metadata.get("workspace_copy") or task.get("workspace_copy") or {}
    file_baseline = metadata.get("file_baseline") or workspace_copy.get("file_baseline") or {}
    file_input_paths: List[str] = []
    try:
        from ouroboros.workspace_file_outputs import changed_file_inputs
        if not isinstance(file_baseline, dict):
            raise ValueError("snapshot file baseline is not a path map")
        file_input_paths = changed_file_inputs(root, file_baseline)
    except (OSError, ValueError) as exc:
        errors.append({"type": "file_capture_failed", "message": f"{type(exc).__name__}: {exc}"})
        file_baseline = {}
    # Confirm binary rows too, keeping unchanged large files out of outputs without index writes.
    cmd = ["git", "diff", "--numstat", "--no-renames", "-z", "--no-ext-diff", "--no-color", base_ref, "--"]
    changed_tracked = numstat_paths(_git_bytes(cmd, root, errors=errors), cmd,
                                   lambda args: _git_bytes(args, root, errors=errors))
    # A tracked WAL index (committed beside its database, or staged in a copy) is shared memory as an
    # untracked one is: never diffed, and never named for an integration to stage over the target's own.
    wal_indexes = {rel: reason for rel in changed_tracked if (reason := _wal_index_reason(root, rel))}
    tracked_excluded.extend({"path": rel, "reason": reason} for rel, reason in wal_indexes.items())
    changed_tracked = [rel for rel in changed_tracked if rel not in wal_indexes]
    diffstat = ""
    untracked = _git_path_list(["git", "ls-files", "-z", "--others", "--exclude-standard"], root, errors)
    # Copied inputs were deliberately never Git additions. Their exact baseline
    # owns unchanged/modified/deleted detection even if the child stages them.
    untracked = [rel for rel in untracked if rel not in file_baseline]
    # v6.52.2: exclude declared ephemeral scratch (run_command/run_script `scratch=[...]`) so a
    # throwaway verification file the agent forgot to delete never leaks into the workspace patch.
    scratch = _declared_scratch(artifact_dir, root)
    base_sizes = baseline_blob_sizes(root, base_ref, errors=errors)
    # A delegated copy's named evidence folder: its owner retains those files beside this
    # capture, so one NEW since the recorded baseline (untracked, staged or committed in the
    # copy) never rides the work product; a path the baseline holds is the project's own.
    evidence = str(metadata.get("host_evidence") or "").strip("/")
    held = {rel for rel in {*untracked, *changed_tracked} if evidence and rel not in base_sizes
            and rel not in file_baseline and (rel == evidence or rel.startswith(evidence + "/"))}
    excluded.extend({"path": rel, "reason": NAMED_EVIDENCE_EXCLUSION} for rel in sorted(held & set(untracked)))
    held_tracked = sorted(held - set(untracked))
    tracked_excluded.extend({"path": rel, "reason": NAMED_EVIDENCE_EXCLUSION} for rel in held_tracked)
    untracked = [rel for rel in untracked if rel not in held]
    changed_tracked = [rel for rel in changed_tracked if rel not in held]
    # git's binary verdict for the whole inventory in one process (two when empty
    # files need their attribute verdict; #1241); the loop below keeps its per-file
    # order and reads the verdict instead of spawning ``git diff --numstat`` per file.
    binary_verdicts = untracked_binary_verdicts(root, binary_verdict_candidates(root, untracked), warnings=diagnostics)
    for rel in untracked:
        if _is_declared_scratch(root, rel, *scratch):
            excluded.append({"path": rel, "reason": "declared ephemeral scratch (v6.52.2)"})
            continue
        sensitive_reason = _sensitive_untracked_reason(rel)
        if sensitive_reason:
            sensitive.append({"path": rel, "reason": sensitive_reason})
            continue
        reason = _patch_exclude_reason(rel) or _wal_index_reason(root, rel)
        if reason:
            excluded.append({"path": rel, "reason": reason})
            continue
        blob_reason = _untracked_blob_exclude_reason(
            root, rel, file_outputs=file_output_paths, warnings=diagnostics, binary_verdicts=binary_verdicts)
        if blob_reason:
            excluded.append({"path": rel, "reason": blob_reason})
            continue
        included_untracked.append(rel)
    captured_inputs: List[str] = []
    for rel in file_input_paths:
        sensitive_reason = _sensitive_untracked_reason(rel) or pem_capture_refusal(root, rel, warnings=diagnostics)
        if sensitive_reason:
            sensitive.append({"path": rel, "reason": sensitive_reason})
            continue
        if reason := _patch_exclude_reason(rel):
            excluded.append({"path": rel, "reason": reason})
            continue
        captured_inputs.append(rel)
        if (root / rel).is_file() and not (root / rel).is_symlink():
            file_output_paths.append(rel)
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

    # Large generated data rides a file artifact, never a giant binary Git patch.
    large_tracked = []
    for rel in changed_tracked:
        if rel in file_baseline:
            continue
        path = root / rel
        present = path.is_file() and not path.is_symlink()
        size = path.stat().st_size if present else 0
        if max(size, base_sizes.get(rel, 0)) > _PATCH_FILE_REFERENCE_BYTES:
            large_tracked.append(rel)
            if present:
                file_output_paths.append(rel)
            tracked_excluded.append({"path": rel, "reason": "captured as file-reference artifact",
                                     "size": size, "base_size": base_sizes.get(rel, 0),
                                     "deleted": not path.exists() and not path.is_symlink(),
                                     "symlink": path.is_symlink(),
                                     "link_target": os.readlink(path) if path.is_symlink() else ""})
    file_outputs: List[Dict[str, Any]] = []
    if file_output_paths and not errors:
        from ouroboros.artifacts import copy_directory_to_task_artifacts
        try:
            file_outputs = copy_directory_to_task_artifacts(
                None, root, kind="workspace_file_outputs", artifact_dir=artifact_dir,
                member_paths=[root / rel for rel in file_output_paths],
            )
        except (OSError, ValueError) as exc:
            errors.append({"type": "file_capture_failed", "message": f"{type(exc).__name__}: {exc}"})
    file_changes: List[Dict[str, Any]] = []
    if (file_output_paths or large_tracked or captured_inputs) and not errors:
        from ouroboros.workspace_file_outputs import capture_file_output_changes
        try:
            file_changes = capture_file_output_changes(
                root, base_ref, [*file_output_paths, *large_tracked, *captured_inputs], file_outputs, artifact_dir,
                file_baseline=file_baseline,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append({"type": "file_capture_failed", "message": f"{type(exc).__name__}: {exc}"})
    total_size = 0
    with patch_path.open("wb") as fh:
        if not errors:
            tracked_lock_excludes = sorted(set(changed_tracked) & incidental_lock_excludes)
            tracked_pathspec = ["--"]
            if tracked_lock_excludes or large_tracked or file_baseline or wal_indexes or held_tracked:
                tracked_pathspec += ["."] + [f":(exclude,literal){rel}" for rel in [
                    *tracked_lock_excludes, *large_tracked, *file_baseline, *wal_indexes, *held_tracked]]
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
                allow_rc={0},
                errors=errors,
                diagnostics=diagnostics,
            )
            for rel in included_untracked:
                if total_size:
                    total_size += _write_patch_separator(fh)
                total_size += _append_git_output(
                    ["git", "diff", "--no-index", "--binary", "--no-ext-diff", "--no-color", "--", os.devnull, rel],
                    root,
                    fh,
                    allow_rc={0, 1},
                    errors=errors,
                    diagnostics=diagnostics,
                )
    head_errors: List[Dict[str, Any]] = []
    current_head = _git_stdout(["git", "rev-parse", "--verify", "HEAD"], root, allow_rc={0}, errors=head_errors).strip()
    # Only isolated copies keep HEAD fixed to their provisioned baseline.
    # Shared external/genesis trees may receive legitimate parent commits;
    # integration verifies their postimages instead of calling that drift.
    if (task_base_sha and acting_constraint is not None
            and acting_constraint.surface == "self_worktree" and current_head != base_head):
        if not current_head:
            errors.extend(head_errors)
        errors.append({
            "type": "workspace_head_changed" if current_head else "workspace_head_unverified",
            "message": ("workspace HEAD changed during task execution; patch artifact is invalid"
                        if current_head else "workspace HEAD could not be verified at artifact finalization"),
            "expected_head": base_head,
            "current_head": current_head or "",
        })
    if errors:
        total_size, digest = 0, ""
        status = ARTIFACT_STATUS_FAILED
    elif total_size > 0 or file_changes:
        status = ARTIFACT_STATUS_READY_WITH_CHANGES
    else:
        status = ARTIFACT_STATUS_READY_NO_CHANGES
    if not total_size:
        try:
            patch_path.unlink()
        except OSError:
            pass
        digest = ""
    else:
        compress_in_place(patch_path)
        measured = stream_artifact_file(patch_path)
        total_size, digest = measured["size"], measured["sha256"]
    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": status,
        "workspace_root": str(root),
        **({"workspace_copy": workspace_copy} if workspace_copy else {}),
        "patch_name": "workspace.patch",
        "manifest_name": "workspace_patch.json",
        "base_provenance": ("task_constraint" if task_base_sha else "admission_head" if preflight_head
                            else "empty_tree" if base_is_empty_tree else "capture_head"),
        "base_explanation": ("Current workspace at capture time relative to the recorded base; may include branch differences and later unrelated changes. It does not attribute authorship."
                             if task_base_sha or preflight_head else
                             "Current workspace comparison at capture time; no original admission basis was recorded. It may include later unrelated changes and does not attribute authorship."),
        "comparison_note": "No auxiliary comparison target was selected. Use vcs_read(action=diff, base=..., head=...) for two trees, or base only for the current worktree.",
        "current_branch": _git_stdout(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], root, allow_rc={0, 1}, errors=diagnostics).strip(),
        "tracking_upstream": _git_stdout(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], root, allow_rc={0, 128}, errors=diagnostics).strip(),
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
        "file_outputs": file_outputs,
        "file_output_changes": file_changes,
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

    # Published with its measured identity, like the patch beside it (#1969).
    artifacts = [
        {
            "kind": "workspace_patch_manifest",
            "name": "workspace_patch.json",
            "path": str(manifest_path),
            **stream_artifact_file(manifest_path),
            "workspace_root": str(root),
        }
    ]
    if status == ARTIFACT_STATUS_READY_WITH_CHANGES and total_size:
        artifacts.insert(0, {
            "kind": "workspace_patch",
            "name": "workspace.patch",
            "path": str(patch_path),
            "size": total_size,
            "sha256": digest,
            "workspace_root": str(root),
        })
    return [*artifacts, *file_outputs], manifest


def _declared_scratch(artifact_dir: pathlib.Path, root: pathlib.Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """The declared scratch of ``artifact_dir``'s manifest as ``({rel: sha256}, {normcased abs: sha256})``.

    The manifest stores {abs_path: sha256}; a file is excluded ONLY while its CURRENT content
    still matches the recorded scratch sha — so a LATER real file written to the same path
    (different content) is NOT dropped. Empty/absent/mismatched => included (no regression).
    """
    by_rel: Dict[str, str] = {}
    by_abs: Dict[str, str] = {}
    try:
        scratch_map = json.loads((artifact_dir / SCRATCH_MANIFEST_NAME).read_text(encoding="utf-8")).get("scratch")
        if isinstance(scratch_map, dict):
            for raw_abs, sha in scratch_map.items():
                try:
                    resolved = pathlib.Path(str(raw_abs)).resolve(strict=False)
                    by_abs[os.path.normcase(str(resolved))] = str(sha)
                    by_rel[resolved.relative_to(root).as_posix()] = str(sha)
                except Exception:
                    continue
    except Exception:
        return {}, {}
    return by_rel, by_abs


def _is_declared_scratch(root: pathlib.Path, rel: str, by_rel: Dict[str, str], by_abs: Dict[str, str]) -> bool:
    """Whether untracked ``rel`` is declared scratch whose current content still has its recorded sha."""
    want_sha = by_rel.get(rel)
    if not want_sha and by_abs:  # resolved only to match a declared scratch path
        try:
            want_sha = by_abs.get(os.path.normcase(str((root / rel).resolve(strict=False))))
        except (OSError, RuntimeError):  # a link that cannot be resolved (a loop) is no scratch file
            want_sha = None
    if not want_sha:
        return False
    try:
        return sha256((root / rel).read_bytes()).hexdigest() == want_sha
    except OSError:
        return False


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

    from ouroboros.workspace_preflight import git_head_basis

    head, state = git_head_basis(root)
    if state == "known":
        return head, head, False
    if state == "unborn":
        empty_tree = _git_empty_tree_oid(root, errors)
        if empty_tree:
            return empty_tree, _GIT_UNBORN_HEAD, True
    errors.append({"type": "git_invalid_head", "message": "HEAD could not be verified; unborn state is unproven"})
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


def baseline_blob_sizes(root: pathlib.Path, base_ref: str, *, errors: List[Dict[str, Any]]) -> Dict[str, int]:
    """The recorded baseline's blobs (symlink entries included), shared by capture and retention."""
    sizes: Dict[str, int] = {}
    for row in _git_bytes(["git", "ls-tree", "-r", "-l", "-z", base_ref, "--"], root, errors=errors).split(b"\0"):
        header, separator, raw_path = row.partition(b"\t")
        fields = header.split()
        if separator and len(fields) == 4 and fields[3].isdigit():
            sizes[raw_path.decode("utf-8", errors="replace")] = int(fields[3])
    return sizes


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
    from ouroboros.git_observation import host_git_env
    try:
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            timeout=30,
            env=host_git_env(),
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
    *,
    allow_rc: set[int],
    errors: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
) -> int:
    from ouroboros.git_observation import host_git_env
    written_box = {"value": 0}
    read_errors: List[str] = []
    try:
        with tempfile.TemporaryFile() as stderr_fh:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                env=host_git_env(),
            )
            assert proc.stdout is not None

            def _reader() -> None:
                try:
                    while True:
                        chunk = proc.stdout.read(1024 * 128)
                        if not chunk:
                            break
                        fh.write(chunk)
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


def _write_patch_separator(fh: BinaryIO) -> int:
    data = b"\n"
    fh.write(data)
    return len(data)


# PEM private-key header (any label: RSA/EC/DSA/OPENSSH/ENCRYPTED/none). This is
# CONTENT evidence — the filename-shape rules in workspace_patch_rules miss real
# key material under an innocent name (notes.txt) while flagging public.pem; a
# bounded head read catches the former on evidence, not on spelling (#447).
_PEM_PRIVATE_KEY_RE = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_PEM_HEAD_READ_BYTES = 4096


def pem_private_key_reason(root: pathlib.Path, rel: str) -> str:
    """Reason a file carries private-key CONTENT, or ``""`` when it does not.

    The bounded head read is the evidence every git lane shares: the patch,
    the cooperative checkpoint and the attached-folder snapshot. Unreadable
    heads are treated as ordinary content (fail-soft: git still decides).
    """
    try:
        with (root / rel).open("rb") as fh:
            head = fh.read(_PEM_HEAD_READ_BYTES)
    except OSError:
        return ""
    return "private key material (PEM private-key header)" if _PEM_PRIVATE_KEY_RE.search(head) else ""


def pem_capture_refusal(root: pathlib.Path, rel: str, *, warnings=None) -> str:
    """Keep the observed PEM finding while the shared effective mode decides its effect."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    reason = pem_private_key_reason(root, rel)
    if reason and mode_has_unrestricted_agency(get_runtime_mode()):
        finding = {"path": rel, "reason": reason, "advisory": True}
        if warnings is not None:
            warnings.append(finding)
        else:
            import logging
            logging.getLogger(__name__).warning("PEM capture finding retained as advisory: %s", finding)
        return ""
    return reason


def _wal_index_reason(root: pathlib.Path, rel: str) -> str:
    """A SQLite WAL index never enters a capture: no raw reader opens it (``store.is_wal_index``:
    its bytes are shared memory under the holders' locks), and a copy of its database rebuilds its
    own from the log at the first open. It is excluded before any read, the PEM head included."""
    from ouroboros.store import is_wal_index

    return "SQLite WAL index (shared memory a database copy rebuilds from its log)" if is_wal_index(root / rel) else ""


def _untracked_blob_exclude_reason(root: pathlib.Path, rel: str, *, file_outputs: Optional[List[str]] = None,
                                   warnings=None, binary_verdicts: Optional[Dict[str, bool]] = None) -> str:
    """Reason to drop an untracked file from the workspace patch when it is a
    build/runtime BINARY, exceeds the per-file size cap, or carries a PEM
    private-key header in its head bytes. Keeps real-usage patches
    source-shaped without losing data (the file stays in the workspace
    and is recorded under ``untracked_excluded``). On any git/stat failure the
    file is INCLUDED (conservative — the main binary diff still applies).

    ``binary_verdicts`` is one :func:`untracked_binary_verdicts` batch over the whole
    inventory; ``None`` keeps git's per-file ``--numstat`` verdict (one subprocess per
    file) for a caller that did not batch."""

    try:
        size = (root / rel).lstat().st_size
    except OSError:
        return ""  # unreadable/symlink races: include and let git decide
    if reason := pem_capture_refusal(root, rel, warnings=warnings):
        return reason
    if size > _PATCH_MAX_UNTRACKED_FILE_BYTES:
        if file_outputs is not None:
            file_outputs.append(rel)
        return f"untracked file exceeds size cap ({size}B > {_PATCH_MAX_UNTRACKED_FILE_BYTES}B)"
    if binary_verdicts is None:
        numstat = _git_stdout(
            ["git", "diff", "--no-index", "--numstat", "--no-ext-diff", "--no-color", "--", os.devnull, rel],
            root,
            allow_rc={0, 1},
            errors=None,
        )
        first = numstat.strip().splitlines()[0] if numstat.strip() else ""
        binary = first.startswith("-\t-")
    else:
        binary = bool(binary_verdicts.get(rel, False))
    if binary:
        if file_outputs is not None:
            file_outputs.append(rel)
        return "binary file"
    return ""


def binary_verdict_candidates(root: pathlib.Path, rels: Sequence[str]) -> List[str]:
    """The untracked paths that still need git's binary verdict.

    The per-file predicate decides the dotenv policy, the name rules, the PEM head
    and the size cap BEFORE it ever asks git; the batch keeps that order, so a
    vetoed or oversized file is never handed to git — its clean filters and
    encodings run only over files that may become Git inputs, exactly as before."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    unrestricted = mode_has_unrestricted_agency(get_runtime_mode())
    out: List[str] = []
    for rel in rels:
        if (not rel or _sensitive_untracked_reason(rel) or _patch_exclude_reason(rel)
                or _wal_index_reason(root, rel)):
            continue
        try:
            info = os.lstat(root / rel)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        if not unrestricted and pem_private_key_reason(root, rel):  # PEM before the size cap, as the predicate orders them
            continue
        if info.st_size > _PATCH_MAX_UNTRACKED_FILE_BYTES:
            continue
        out.append(rel)
    return out


def untracked_binary_verdicts(root: pathlib.Path, rels: Sequence[str], *, warnings=None) -> Optional[Dict[str, bool]]:
    """git's own binary verdict for every path in ``rels`` from ONE ``git diff``.

    Every REGULAR file is staged as the empty blob into a scratch index
    (``update-index --index-info`` touches no file and writes no content object), and
    one index-versus-worktree ``git diff --numstat -z`` then runs exactly the machinery
    a per-file ``git diff --no-index --numstat`` ran — attributes, diff drivers, clean
    filters and working-tree encodings included — so the answer is git's, not a
    re-implementation (parity probed on git 2.53 for 19 path classes). ``-\\t-`` is
    binary; a text file and a file that vanished meanwhile (``0\\t0``) are text, exactly
    as the per-file verdict reads them. A path the diff OMITS is identical to the staged
    blob — an empty file — which git still classifies by attribute (an empty ``*.dat``
    under ``-diff`` is binary): those are asked once more, staged as a one-byte blob, so
    the empty-file set costs a second process, never one per file. Symlinks and other
    non-regular paths are text (git diffs the link text). One inventory of tens of
    thousands of files therefore costs one or two processes (#1241). ``None`` — the
    "did not batch" signal — when git cannot answer, after an advisory warning: the
    callers keep the per-file verdict (the same answer, one process per file), never a
    new refusal."""
    regular: List[str] = []
    for rel in rels:
        try:
            if rel and stat.S_ISREG(os.lstat(root / rel).st_mode):
                regular.append(rel)
        except OSError:
            continue
    if not regular:
        return {}
    env = dict(os.environ)
    scratch = ""

    def _git(*args: str, data: bytes = b"") -> bytes:
        return subprocess.run(["git", *args], cwd=str(root), capture_output=True, input=data,
                              env=env, timeout=300, check=True).stdout

    def _diff_against(staged: bytes, paths: List[str]) -> Dict[str, bool]:
        """One index-versus-worktree diff with every path staged as ``staged``: the
        verdict of each path that produced a row (absent = identical to ``staged``)."""
        blob = _git("hash-object", "-w", "--stdin", data=staged).strip()
        _git("read-tree", "--empty")
        _git("update-index", "-z", "--index-info",
             data=b"".join(b"100644 " + blob + b"\t" + os.fsencode(rel) + b"\0" for rel in paths))
        seen: Dict[str, bool] = {}
        for row in _git("diff", "--numstat", "-z", "--no-renames", "--no-ext-diff", "--no-color").split(b"\0"):
            added, sep, rest = row.partition(b"\t")
            deleted, sep2, path = rest.partition(b"\t")
            if sep and sep2:
                seen[os.fsdecode(path)] = added == b"-" and deleted == b"-"
        return seen

    try:
        fd, scratch = tempfile.mkstemp(prefix="ouroboros-binary-verdict-", suffix=".index")
        os.close(fd)
        env["GIT_INDEX_FILE"] = scratch
        verdicts = _diff_against(b"", regular)
        omitted = [rel for rel in regular if rel not in verdicts]
        if omitted:  # empty files: a second batch, staged as a one-byte blob
            verdicts.update(_diff_against(b"\n", omitted))
    except Exception as exc:
        if warnings is not None:
            warnings.append({"reason": "binary_verdict_batch_unavailable", "advisory": True,
                             "detail": f"{type(exc).__name__}: {exc}"[:300]})
        return None
    finally:
        if scratch:
            try:
                os.unlink(scratch)
            except OSError:
                pass
    return {rel: verdicts.get(rel, False) for rel in regular}


def untracked_capture_veto_reason(root: pathlib.Path, rel: str, *, file_outputs: Optional[List[str]] = None,
                                  warnings=None, binary_verdicts: Optional[Dict[str, bool]] = None) -> str:
    """Classify an untracked file for Git, file-reference transfer, or exclusion.

    The delegated-run baseline snapshot
    (``subagent_worktrees.provision_execution_snapshot``) asks the SAME three
    checks, in the SAME order, that ``write_workspace_patch_artifacts`` applies
    to untracked files: the dotenv spellings and exact credential leaves first,
    then the static junk rules, then the blob veto, which reads the head bytes
    for a PEM private-key header before the size cap and the binary check. One
    combined predicate here so the snapshot and the patch cannot drift apart
    about eligibility.
    Returns the human-readable patch exclusion reason, or "" for a Git input.
    ``file_outputs`` receives eligible binary/large files which use file
    artifacts rather than Git blobs; credential/junk exclusions never enter it.
    ``binary_verdicts`` comes from one :func:`untracked_binary_verdicts` batch over
    the whole inventory (``None`` = per-file git verdict, see the blob check).
    """
    reason = _sensitive_untracked_reason(rel)
    if reason:
        return reason
    reason = _patch_exclude_reason(rel) or _wal_index_reason(root, rel)
    if reason:
        return reason
    return _untracked_blob_exclude_reason(root, rel, file_outputs=file_outputs, warnings=warnings,
                                          binary_verdicts=binary_verdicts)


def _preflight_head_from_task(task: Dict[str, Any]) -> str:
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    return str(git.get("head") or (_GIT_UNBORN_HEAD if git.get("head_state") == "unborn" else ""))


def _preflight_head_present(task: Dict[str, Any]) -> bool:
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return "workspace_preflight" in meta


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
