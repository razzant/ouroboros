"""Low-level git plumbing shared by the git tool owners.

Runtime-mode projection, git error sanitisation and structured result
publication, staging hygiene, the cross-process git lock, and the
resolved-binding path projections that every git tool leaf builds on.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import List

from ouroboros.config import get_runtime_mode
from ouroboros.runtime_mode_policy import format_protected_paths
from ouroboros.platform_layer import acquire_exclusive_file_lock, unlink_lockfile
from ouroboros.tools.registry import ToolContext, system_repo_dir_for
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from ouroboros.utils import utc_now_iso, write_text, run_cmd


def _current_runtime_mode() -> str:
    try:
        return get_runtime_mode()
    except Exception:
        return "advanced"


def _protected_paths_block_message(paths, *, runtime_mode: str, action: str) -> str:
    rendered = format_protected_paths(paths)
    return (
        f"⚠️ CORE_PROTECTION_BLOCKED: runtime_mode={runtime_mode!r} refuses "
        f"to {action} protected Ouroboros core/contract/release path(s): {rendered}. "
        "Use runtime_mode='pro' and pass the normal triad + scope review before "
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


def _publish_review_blocked(ctx: ToolContext, text: str) -> str:
    """Publish one reviewer-finding rejection without relabelling other blocks."""
    return _publish_tool_result(
        ctx,
        ToolResult(status="ok", code="REVIEW_BLOCKED", text=text),
    )


_BINARY_EXTENSIONS = frozenset({
    ".so", ".dylib", ".dll", ".a", ".lib", ".o", ".obj",
    ".pyc", ".pyo", ".whl", ".egg",
})

def _ensure_gitignore(repo_dir) -> None:
    gi = pathlib.Path(repo_dir) / ".gitignore"
    if not gi.exists():
        write_text(gi, "__pycache__/\n*.pyc\n*.pyo\n*.so\n*.dylib\n*.dll\n"
                       "*.dist-info/\nbase_library.zip\n.DS_Store\n")  # atomic (G)
def _unstage_binaries(repo_dir) -> List[str]:
    try:
        staged = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_dir)
    except Exception:
        return []
    removed = []
    for f in staged.strip().splitlines():
        f = f.strip()
        if f and pathlib.Path(f).suffix.lower() in _BINARY_EXTENSIONS:
            try:
                run_cmd(["git", "reset", "HEAD", "--", f], cwd=repo_dir)
                removed.append(f)
            except Exception:
                pass
    return removed


def _acquire_git_lock(ctx: ToolContext, timeout_sec: int = 120) -> pathlib.Path:
    lock_dir = ctx.drive_path("locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "git.lock"
    fd = acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=float(timeout_sec),
        stale_sec=600.0,
        metadata=f"locked_at={utc_now_iso()}\n",
        poll_sec=0.5,
    )
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return lock_path
    raise TimeoutError(f"Git lock not acquired within {timeout_sec}s: {lock_path}")


def _release_git_lock(lock_path: pathlib.Path) -> None:
    unlink_lockfile(lock_path)


def _binding_repo_rel(binding: ResolvedResourceBinding) -> str:
    return binding.target_path.relative_to(binding.base_path).as_posix()


def _binding_targets_system_repo(ctx: ToolContext, binding: ResolvedResourceBinding) -> bool:
    return binding.base_path.resolve(strict=False) == system_repo_dir_for(ctx).resolve(strict=False)
