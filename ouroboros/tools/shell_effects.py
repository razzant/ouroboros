"""What a command did to its working tree, and which of it was throwaway.

Owns the git-worktree discovery and status/diff projections, the bounded
shallow listing and tree fingerprints used to notice a filesystem effect in a
non-git cwd, the effect gate behind the user_files artifact nudge, the
protected-runtime dirty/restore pair, and the declared ``scratch`` lifecycle
(confinement and git-untracked preconditions plus the sha fingerprints that let
workspace patch capture exclude an ephemeral verification file). The handlers
that call this surface stay with ``tools/shell.py``.
"""

from __future__ import annotations

import hashlib
from hashlib import sha256
import os
import pathlib
import stat
import subprocess
from typing import List

from ouroboros.artifacts import record_task_scratch
from ouroboros.runtime_mode_policy import (
    is_protected_runtime_path,
)
from ouroboros.tools.registry import (
    ToolContext,
)
from ouroboros.tool_access import (
    path_is_relative_to,
)
from ouroboros.utils import safe_relpath


def _resolve_git_root(path: pathlib.Path) -> pathlib.Path | None:
    try:
        from ouroboros.review_state import discover_repo_root
        root = discover_repo_root(path)
        if not (root / ".git").exists():
            return None
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return root if probe.returncode == 0 and probe.stdout.strip() == "true" else None
    except Exception:
        return None


def _status_snapshot(repo_dir: pathlib.Path | None) -> list[str]:
    if repo_dir is None:
        return []
    return sorted(_get_changed_files(repo_dir))


def _shallow_listing(work_dir: pathlib.Path, cap: int = 5000) -> dict:
    """Bounded immediate-children {name: (mtime_ns, size)} snapshot of a cwd. One
    directory level, capped — NOT a recursive filesystem monitor (R5). Used to
    detect a non-git user_files cwd actually producing a top-level deliverable."""
    out: dict = {}
    try:
        with os.scandir(work_dir) as it:
            for entry in it:
                if len(out) >= cap:
                    break
                try:
                    st = entry.stat(follow_symlinks=False)
                    out[entry.name] = (int(st.st_mtime_ns), int(st.st_size))
                except OSError:
                    continue
    except OSError:
        return {}
    return out


def _user_files_run_had_effect(
    before_changed: list[str],
    after_changed: list[str],
    before_listing: dict | None,
    work_dir: pathlib.Path,
) -> bool:
    """Effect-based gate for the ARTIFACT_AUDIT_GAP nudge (R5): warn only when the
    command produced an OBSERVABLE filesystem change in the cwd, not merely
    because it ran in a user_files cwd. Git-tracked cwd (e.g. dig-direct /app) →
    a status delta (modified or new untracked file). Non-git cwd → a bounded
    shallow immediate-children snapshot delta. A read-only command (ls/cat/grep)
    changes neither and is no longer falsely flagged."""
    if after_changed != before_changed:
        return True
    if before_listing is not None:
        return _shallow_listing(work_dir) != before_listing
    return False


def _protected_runtime_dirty_paths(repo_dir: pathlib.Path) -> list[str]:
    dirty: set[str] = set()
    for cmd in (["git", "diff", "--name-only"], ["git", "diff", "--cached", "--name-only"]):
        try:
            res = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                dirty.update(rel for rel in res.stdout.splitlines() if is_protected_runtime_path(rel))
        except Exception:
            pass
    return sorted(dirty)


def _restore_protected_runtime_paths(repo_dir: pathlib.Path, paths: list[str]) -> list[str]:
    restored: list[str] = []
    for rel in sorted(set(paths)):
        try:
            subprocess.run(
                ["git", "reset", "HEAD", "--", rel],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=5,
            )
            subprocess.run(
                ["git", "checkout", "--", rel],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=5,
            )
            restored.append(rel)
        except Exception:
            pass
    return restored


def _tree_fingerprint(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    root = pathlib.Path(path)
    if not root.exists():
        return ""
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            st = child.lstat()
        except OSError:
            continue
        try:
            rel = child.relative_to(root).as_posix()
        except ValueError:
            rel = safe_relpath(str(child))
        digest.update(rel.encode("utf-8", errors="replace"))
        digest.update(str(st.st_mode).encode())
        digest.update(str(st.st_size).encode())
        digest.update(str(st.st_mtime_ns).encode())
        if stat.S_ISLNK(st.st_mode):
            try:
                digest.update(os.readlink(child).encode("utf-8", errors="replace"))
            except OSError:
                pass
    return digest.hexdigest()


def _resolve_scratch_abs(scratch: List[str] | None, work_dir) -> list[pathlib.Path]:
    """Resolve declared ephemeral `scratch=[...]` paths to absolute host paths (relative ones
    against the command cwd). Blank entries dropped. (v6.52.2)"""
    base = pathlib.Path(work_dir).resolve(strict=False) if work_dir else None
    out: list[pathlib.Path] = []
    for raw in (scratch or []):
        text = str(raw or "").strip()
        if not text:
            continue
        p = pathlib.Path(text).expanduser()
        out.append((p if p.is_absolute() else ((base / p) if base is not None else p)).resolve(strict=False))
    return out


def _scratch_safety_reason(ctx: ToolContext, scratch_abs: list[pathlib.Path], work_dir, repo_root) -> str:
    """Pre-exec gate for declared scratch (v6.52.2; v6.56.0 adoptable): the cwd must be inside a git
    worktree (so the git-untracked proof is meaningful and the patch-exclusion contract applies), and
    each path must be CONFINED to the command cwd and git-UNTRACKED — so an ephemeral verification
    file can never mask a real TRACKED edit. Returns a refusal reason or ''.

    v6.56.0: a path is no longer blocked merely because it already EXISTS. Re-declaring the same
    throwaway across commands, or adopting an untracked file created earlier in THIS task (e.g. via
    write_file, or a prior command), is a normal verification loop — the git-tracked check still
    blocks masking a real edit, and headless patch exclusion stays sha-gated (a later real rewrite
    diverges the sha and is NOT dropped). On adoption we record the current sha through the SSOT
    writer so the manifest reflects the adopted state at declaration time."""
    if not scratch_abs:
        return ""
    if repo_root is None:
        # No git worktree at the cwd: we cannot prove a path is git-untracked, and there is no
        # workspace patch to exclude it from — so scratch is not meaningful here.
        return "scratch requires a git-worktree cwd (it is for in-repo verification); use outputs= for a deliverable"
    base = pathlib.Path(work_dir).resolve(strict=False) if work_dir else None
    tracked: set[str] = set()
    try:
        res = subprocess.run(["git", "ls-files"], cwd=str(repo_root), capture_output=True, text=True, timeout=20)
        if res.returncode == 0:
            root = pathlib.Path(repo_root).resolve(strict=False)
            tracked = {str((root / line.strip()).resolve(strict=False)) for line in (res.stdout or "").splitlines() if line.strip()}
    except Exception:
        tracked = set()
    adopt: dict = {}
    for cand in scratch_abs:
        if base is not None and not (cand == base or path_is_relative_to(cand, base)):
            return f"scratch path escapes the command cwd ({base}): {cand}"
        if str(cand) in tracked:
            return f"scratch path is git-tracked — not a throwaway (use outputs=, or edit it as a real change): {cand}"
        # A directory can neither be sha-fingerprinted nor excluded from the patch
        # file-by-file — silently adopting one would let its contents leak into the
        # deliverable while SCRATCH_REMAINS nags forever. Refuse explicitly.
        try:
            if cand.is_dir():
                return f"scratch path is a directory — declare the throwaway FILES, not their parent dir: {cand}"
        except OSError:
            pass
        # Adoptable: an existing untracked+confined file — record its current sha now so a
        # re-declaration is idempotent and the adopted state is captured at declaration.
        try:
            if cand.is_file():
                adopt[str(cand)] = sha256(cand.read_bytes()).hexdigest()
        except OSError:
            continue
    if adopt:
        record_task_scratch(ctx, adopt)
    return ""


def _record_scratch_fingerprints(ctx: ToolContext, scratch_abs: list[pathlib.Path]) -> None:
    """Record sha256 of declared scratch files that exist NOW (post-exec) so workspace patch
    capture can exclude them while they still match. Called on EVERY exit path — normal, nonzero,
    timeout, and exception — so a file created by a command that then times out is still managed
    (v6.52.2). Fail-soft; only records files that currently exist."""
    if not scratch_abs:
        return
    fingerprints: dict = {}
    for sp in scratch_abs:
        try:
            if sp.is_file():
                fingerprints[str(sp)] = sha256(sp.read_bytes()).hexdigest()
        except OSError:
            continue
    if fingerprints:
        record_task_scratch(ctx, fingerprints)


def _get_changed_files(repo_dir: pathlib.Path) -> list:
    """Return changed files after an edit."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return [line[3:].strip() for line in res.stdout.splitlines() if len(line) > 3 and line.strip()]
    except Exception:
        pass
    return []


def _get_diff_stat(repo_dir: pathlib.Path) -> str:
    """Return git diff --stat output."""
    try:
        res = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""
