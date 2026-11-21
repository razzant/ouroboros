"""Low-level git plumbing shared by the git tool owners.

Runtime-mode projection, git error sanitisation, staging hygiene, the
cross-process git lock, and resolved-binding path projections live here.
The ``ouroboros/tools/git.py`` facade re-exports these definitions. Shared
helpers are read through the call-time handle ``_git()`` so the facade binding
stays the one tests monkeypatch.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from typing import Any, Dict, List, NamedTuple, Tuple

from ouroboros.tools.registry import ToolContext
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.utils import utc_now_iso


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import git

    return git


def _current_runtime_mode() -> str:
    try:
        return _git().get_runtime_mode()
    except Exception:
        return "advanced"


def _protected_paths_block_message(paths, *, runtime_mode: str, action: str) -> str:
    rendered = _git().format_protected_paths(paths)
    return (
        f"⚠️ CORE_PROTECTION_BLOCKED: runtime_mode={runtime_mode!r} refuses "
        f"to {action} protected Ouroboros core/contract/release path(s): {rendered}. "
        "Use runtime_mode='pro' or 'cyber_pro' and pass the normal panel review before "
        "committing protected surfaces."
    )


def _sanitize_git_error(msg: str) -> str:
    return re.sub(r"(https?://)([^@\s]+@)", r"\1<redacted>@", msg)


def _publish_git_error(ctx: ToolContext, text: str) -> str:
    """Publish one structurally known Git terminal without changing public text."""
    return _publish_tool_result(
        ctx,
        ToolResult(status="ok", code="GIT_ERROR", text=text),
    )


_BINARY_EXTENSIONS = frozenset({
    ".so", ".dylib", ".dll", ".a", ".lib", ".o", ".obj",
    ".pyc", ".pyo", ".whl", ".egg",
})


def _ensure_gitignore(repo_dir) -> None:
    gi = pathlib.Path(repo_dir) / ".gitignore"
    if not gi.exists():
        _git().write_text(gi, "__pycache__/\n*.pyc\n*.pyo\n*.so\n*.dylib\n*.dll\n"
                       "*.dist-info/\nbase_library.zip\n.DS_Store\n")  # atomic (G)


def _staged_index(repo_dir) -> Dict[str, str]:
    """Every index path that differs from HEAD, rename sides apart, with its status letter."""
    fields = _git().run_cmd(["git", "diff", "--cached", "--no-renames", "--name-status", "-z"],
                            cwd=repo_dir).split("\0")
    return dict(zip(fields[1::2], fields[0::2]))


WHOLE_INDEX = "whole index"  # ``_staged_selection`` after a whole-tree ``git add -A``


def _unstage_candidate(ctx: ToolContext) -> None:
    """Undo exactly this call's staging (``ctx._staged_selection``, recorded by
    ``_stage_candidate_for_review``): the recorded paths, so a path someone else staged
    stays staged; the whole index after a whole-tree add; nothing before any staging.
    Both are path resets: a pathless ``git reset HEAD`` would also drop a staged merge
    (MERGE_HEAD, MERGE_MODE, MERGE_MSG). ``:/`` names the whole tree from any directory."""
    paths = getattr(ctx, "_staged_selection", None)
    if paths == WHOLE_INDEX:
        _git().run_cmd(["git", "reset", "-q", "HEAD", "--", ":/"], cwd=ctx.repo_dir)
    elif paths:
        _git().run_cmd(["git", "--literal-pathspecs", "reset", "-q", "HEAD", "--", *paths], cwd=ctx.repo_dir)


def _unstage_binaries(repo_dir) -> List[str]:
    try:
        staged = _git().run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_dir)
    except Exception:
        return []
    removed = []
    for f in staged.strip().splitlines():
        f = f.strip()
        if f and pathlib.Path(f).suffix.lower() in _git()._BINARY_EXTENSIONS:
            try:
                _git().run_cmd(["git", "reset", "HEAD", "--", f], cwd=repo_dir)
                removed.append(f)
            except Exception:
                pass
    return removed


class GitLock(NamedTuple):
    """A held working-tree git lock: its file and the descriptor that keeps the kernel hold until release."""
    path: pathlib.Path
    fd: int


def _git_lock_path(ctx: ToolContext, repo_dir: Any = None) -> pathlib.Path:
    """One lock per working tree's own git directory (a linked worktree has its own): copies of the body never
    wait for each other, while one tree's index and HEAD stay serialized (audit of early stops, 06.11)."""
    repo = pathlib.Path(repo_dir or ctx.repo_dir)
    try:
        from ouroboros.work_domain import run

        git_dir = run(["git", "rev-parse", "--absolute-git-dir"], cwd=str(repo), capture_output=True,
                      text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_dir = ""  # not a work tree: its own path keys the lock
    key = hashlib.sha256(os.path.realpath(git_dir or str(repo)).encode("utf-8")).hexdigest()[:16]
    return ctx.drive_path("locks") / f"git-{key}.lock"


def _git_lock_holder(lock_path: pathlib.Path) -> str:
    try:
        fields = dict(item.split("=", 1) for item in lock_path.read_text(encoding="utf-8").split() if "=" in item)
    except (OSError, ValueError):
        return "holder unknown"
    return f"task {fields.get('task', '?')}, pid {fields.get('pid', '?')}, since {fields.get('locked_at', '?')}"


def _acquire_git_lock(ctx: ToolContext, timeout_sec: int = 120, *, repo_dir: Any = None) -> GitLock:
    """Hold this working tree's git lock until ``_release_git_lock``: the descriptor keeps its kernel hold and
    its pid names a live owner, so a waiter never evicts a live holder, only a dead one (owner-aware stale)."""
    lock_path = _git_lock_path(ctx, repo_dir)
    fd = _git().acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=float(timeout_sec),
        stale_sec=600.0,
        metadata=f"pid={os.getpid()} task={getattr(ctx, 'task_id', '') or '-'} locked_at={utc_now_iso()}\n",
        poll_sec=0.5,
        owner_aware_stale=True,
    )
    if fd is not None:
        return GitLock(lock_path, fd)
    raise TimeoutError(f"Git lock not acquired within {timeout_sec}s: this working tree is busy with another git step "
                       f"({_git_lock_holder(lock_path)}); nothing was staged, reviewed or committed by this call. "
                       f"Retry the same call after that step ends. Lock: {lock_path}")


def _release_git_lock(lock: Any) -> None:
    if isinstance(lock, GitLock):
        from ouroboros.platform_layer import release_exclusive_file_lock
        release_exclusive_file_lock(lock.path, lock.fd)
    elif lock is not None:  # a path-only lock of an older caller
        _git().unlink_lockfile(lock)


def _binding_repo_rel(binding: ResolvedResourceBinding) -> str:
    return binding.target_path.relative_to(binding.base_path).as_posix()


def _binding_targets_system_repo(ctx: ToolContext, binding: ResolvedResourceBinding) -> bool:
    return binding.base_path.resolve(strict=False) == _git().system_repo_dir_for(ctx).resolve(strict=False)


def network_git_bounded(args: List[str], *, cwd: Any) -> Tuple[int, str, str]:
    """``update_source._git_network_bounded`` (its bounds, environment and timeout reply) for a service repository;
    a work repository's remote, credentials and ssh are the work domain's own, so there it runs as the work domain."""
    from ouroboros import work_domain
    from supervisor.update_source import FETCH_TIMEOUT_RC, _git_network_bounded

    if not work_domain.is_work(cwd):
        return _git_network_bounded(args, cwd=cwd)
    from ouroboros.update_channels import get_managed_update_fetch_timeout_sec

    limit = float(get_managed_update_fetch_timeout_sec())
    try:
        done = work_domain.run(["git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30", *args],
                               cwd=str(cwd), capture_output=True, text=True, errors="replace", timeout=limit,
                               env={**os.environ, "LC_ALL": "C", "LANG": "C", "GIT_TERMINAL_PROMPT": "0"})
    except subprocess.TimeoutExpired:
        return FETCH_TIMEOUT_RC, "", f"git {' '.join(args[:2])} exceeded {limit:.0f}s and was terminated"
    return done.returncode, (done.stdout or "").strip(), (done.stderr or "").strip()
