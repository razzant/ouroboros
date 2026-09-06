"""Workspace patch capture: the patch artifact, its manifest, and its git plumbing.

Owns the streamed `workspace.patch` and `workspace_patch.json` pair — patch
baseline resolution (including the unborn-HEAD empty-tree case and the acting
subagent `base_sha` binding), the bounded git process helpers the capture runs
on, the declared-scratch and untracked eligibility filtering, the moved-HEAD
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
import subprocess
import tempfile
import threading
from hashlib import sha256
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.headless_status import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.workspace_patch_rules import (
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
