"""Generic VCS inspection and rollback surface for the selected repository.

Owns the vcs_status/vcs_diff/vcs_pull_ff/vcs_restore/vcs_revert behaviour:
binding selection for the generic VCS tools, output limiting, the
fast-forward pull, and the protected-path refusals that keep direct rollback
out of the reviewed-commit lane. The tool descriptors stay with
``tools/git.py``.
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Optional

from ouroboros.runtime_mode_policy import (
    format_protected_paths,
    is_protected_runtime_path,
    normalize_repo_path,
    protected_paths_in,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    binding_targets_system_repo,
    build_resolved_resource_binding,
)
from ouroboros.utils import safe_relpath, run_cmd
from ouroboros.tools.review_helpers import (
    paths_from_porcelain_line as _review_paths_from_porcelain_line,
)
from ouroboros.tools.git_plumbing import (
    _acquire_git_lock,
    _publish_git_error,
    _release_git_lock,
    _sanitize_git_error,
)


def _limit_git_output(text: str, max_chars: int = 0) -> str:
    limit = int(max_chars or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n⚠️ OUTPUT_TRUNCATED: git output limited to {limit} characters by max_chars."


def _vcs_binding(
    ctx: ToolContext,
    binding: Optional[ResolvedResourceBinding],
    *,
    root: str = "system_repo",
    path: str = ".",
) -> ResolvedResourceBinding:
    """Return the dispatch binding, with a system-repo fallback for direct callers.

    Public Tool API calls always receive a registry-built binding. The fallback
    preserves the historical system-repo target for internal/direct helper calls
    without making handler-local target selection part of the public contract.
    """

    resolved = binding or build_resolved_resource_binding(
        ctx,
        root=root,
        operation="vcs",
        path=path or ".",
    )
    if resolved.operation != "vcs" or resolved.root not in {"active_workspace", "system_repo"}:
        raise ValueError(
            "generic VCS tools require a vcs binding on active_workspace or system_repo"
        )
    return resolved


def _vcs_result(text: str, binding: ResolvedResourceBinding) -> str:
    receipt = f"VCS target: root={binding.root}; repo={binding.base_path}"
    rendered = str(text or "").rstrip()
    return f"{rendered}\n\n{receipt}" if rendered else receipt


def _binding_relative_path(binding: ResolvedResourceBinding, requested: str) -> str:
    if not str(requested or "").strip():
        return ""
    try:
        relative = binding.target_path.relative_to(binding.base_path)
    except ValueError as exc:
        raise ValueError("VCS path escapes the selected repository") from exc
    return str(relative) if str(relative) != "." else ""


def _git_status(
    ctx: ToolContext,
    path: str = "",
    max_chars: int = 0,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        cmd = ["git", "status", "--porcelain"]
        if relative := _binding_relative_path(binding, path):
            cmd.extend(["--", safe_relpath(relative)])
        return _vcs_result(
            _limit_git_output(run_cmd(cmd, cwd=binding.base_path), max_chars),
            binding,
        )
    except Exception as e:
        return _publish_git_error(
            ctx,
            f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}",
        )


def _git_diff(
    ctx: ToolContext,
    staged: bool = False,
    path: str = "",
    stat: bool = False,
    name_only: bool = False,
    max_chars: int = 0,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        repo_dir = binding.base_path
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if name_only:
            cmd.append("--name-only")
        elif stat:
            cmd.append("--stat")
        if relative := _binding_relative_path(binding, path):
            cmd.extend(["--", safe_relpath(relative)])
        from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason

        protected_block = protected_artifact_shell_block_reason(
            ctx, cmd, cwd=str(repo_dir), default_cwd=repo_dir, binding=binding,
        )
        if protected_block:
            return _vcs_result(protected_block, binding)
        return _vcs_result(_limit_git_output(run_cmd(cmd, cwd=repo_dir), max_chars), binding)
    except Exception as e:
        return _publish_git_error(
            ctx,
            f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}",
        )


def _ff_pull(repo_dir: pathlib.Path) -> str:
    try:
        branch = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not determine current branch: {e}"
    if not branch or branch == "HEAD":
        return "⚠️ PULL_ERROR: Not on a named branch (detached HEAD). Cannot pull."
    try:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
    except Exception as e:
        return f"⚠️ PULL_ERROR: git fetch failed: {_sanitize_git_error(str(e))}"
    try:
        before_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
        remote_sha = run_cmd(
            ["git", "rev-parse", f"origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not resolve SHAs: {e}"
    if before_sha == remote_sha:
        return f"Already up to date. HEAD={before_sha[:8]} matches origin/{branch}."
    try:
        new_commits = run_cmd(
            ["git", "log", "--oneline", f"HEAD..origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception:
        new_commits = "(could not list commits)"
    try:
        run_cmd(["git", "merge", "--ff-only", f"origin/{branch}"], cwd=repo_dir)
    except Exception as e:
        err = str(e).strip()
        if "Not possible to fast-forward" in err or "diverged" in err.lower():
            return (
                f"⚠️ PULL_ERROR: Branches have diverged — cannot fast-forward.\n"
                f"Local HEAD: {before_sha[:8]}, origin/{branch}: {remote_sha[:8]}\n"
                "Manual resolution needed."
            )
        return f"⚠️ PULL_ERROR: git merge --ff-only failed: {_sanitize_git_error(err)}"
    try:
        after_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
    except Exception:
        after_sha = remote_sha
    lines = [
        f"Pulled origin/{branch}: {before_sha[:8]} → {after_sha[:8]}",
        "", "New commits:",
    ]
    for line in (new_commits or "(none)").splitlines():
        lines.append(f"  {line}")
    return "\n".join(lines)


def _pull_from_remote(
    ctx: ToolContext,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
        return _vcs_result(_ff_pull(binding.base_path), binding)
    except Exception as e:
        return f"⚠️ PULL_ERROR: {_sanitize_git_error(str(e))}"


def _restore_to_head(ctx: ToolContext, confirm: bool = False,
                     paths: Optional[List[str]] = None,
                     root: str = "system_repo",
                     _resolved_binding: Optional[ResolvedResourceBinding] = None) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ RESTORE_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception as e:
        return _vcs_result(f"⚠️ RESTORE_ERROR: git status failed: {e}", binding)
    if not status:
        return _vcs_result("Nothing to restore — working directory is already clean.", binding)
    dirty_files = [
        path
        for line in status.splitlines()
        for path in _review_paths_from_porcelain_line(line)
    ]
    targets_system = binding_targets_system_repo(ctx, binding)
    affected_protected = protected_paths_in(dirty_files) if targets_system else []
    if paths and targets_system:
        for p in paths:
            norm = normalize_repo_path(p)
            if is_protected_runtime_path(norm):
                return _vcs_result(
                    f"⚠️ RESTORE_BLOCKED: Cannot restore protected file: {norm}. "
                    "Protected core/contract/release paths must be changed through reviewed commits.",
                    binding,
                )
    elif affected_protected:
        return _vcs_result(
            f"⚠️ RESTORE_BLOCKED: Uncommitted changes touch protected file(s): "
            f"{format_protected_paths(affected_protected)}. "
            f"Use paths= to restore specific non-critical files, or resolve manually.",
            binding,
        )
    if not confirm:
        try:
            diff_stat = run_cmd(["git", "diff", "--stat"], cwd=repo_dir).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        try:
            untracked = run_cmd(
                ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_dir,
            ).strip()
        except Exception:
            untracked = ""
        preview = ["Uncommitted changes that will be lost:", "", diff_stat]
        if untracked:
            preview.append("")
            preview.append("Untracked files that will be removed:")
            for f in untracked.splitlines()[:15]:
                preview.append(f"  {f}")
        preview.append("")
        preview.append("Call again with confirm=true to proceed.")
        return _vcs_result("\n".join(preview), binding)
    if paths:
        safe_paths = [os.path.normpath(p.strip().lstrip("./")) for p in paths if p.strip()]
        if not safe_paths:
            return _vcs_result("⚠️ RESTORE_ERROR: No valid paths provided.", binding)
        try:
            run_cmd(["git", "checkout", "HEAD", "--"] + safe_paths, cwd=repo_dir)
        except Exception as e:
            return _vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            run_cmd(["git", "clean", "-fd", "--"] + safe_paths, cwd=repo_dir)
        except Exception:
            pass
        return _vcs_result(f"Restored {len(safe_paths)} path(s) to HEAD.", binding)
    else:
        try:
            run_cmd(["git", "checkout", "HEAD", "--", "."], cwd=repo_dir)
        except Exception as e:
            return _vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            run_cmd(["git", "clean", "-fd"], cwd=repo_dir)
        except Exception:
            pass
        return _vcs_result(
            "All uncommitted changes discarded. Working directory matches HEAD.",
            binding,
        )


def _revert_commit(
    ctx: ToolContext,
    sha: str,
    confirm: bool = False,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ REVERT_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    sha = sha.strip()
    if not sha:
        return _vcs_result("⚠️ REVERT_ERROR: sha parameter is required.", binding)
    try:
        full_sha = run_cmd(
            ["git", "rev-parse", "--verify", sha], cwd=repo_dir,
        ).strip()
    except Exception:
        return _vcs_result(f"⚠️ REVERT_ERROR: Commit '{sha}' not found.", binding)
    try:
        parents = run_cmd(
            ["git", "rev-list", "--parents", "-1", full_sha], cwd=repo_dir,
        ).strip().split()
    except Exception:
        parents = [full_sha]
    if len(parents) > 2:
        return _vcs_result(
            f"⚠️ REVERT_ERROR: Commit {sha[:8]} is a merge commit ({len(parents)-1} parents). "
            "git revert on merge commits requires specifying a parent.",
            binding,
        )
    try:
        changed_files = run_cmd(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", full_sha],
            cwd=repo_dir,
        ).strip().splitlines()
    except Exception:
        changed_files = []
    protected_changes = (
        protected_paths_in(changed_files)
        if binding_targets_system_repo(ctx, binding)
        else []
    )
    if protected_changes:
        return _vcs_result(
            f"⚠️ REVERT_BLOCKED: Commit {sha[:8]} touches protected file(s): "
            f"{format_protected_paths(protected_changes)}. "
            "Direct vcs_revert cannot create protected-path commits; stage the intended "
            "revert manually and use commit_reviewed so the normal triad + scope review covers it.",
            binding,
        )
    try:
        commit_msg = run_cmd(
            ["git", "log", "-1", "--format=%s", full_sha], cwd=repo_dir,
        ).strip()
    except Exception:
        commit_msg = "(unknown)"
    if not confirm:
        try:
            diff_stat = run_cmd(
                ["git", "diff", f"{full_sha}^..{full_sha}", "--stat"], cwd=repo_dir,
            ).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        return _vcs_result(
            f"This will revert commit {full_sha[:8]}:\n"
            f"  Message: {commit_msg}\n"
            f"  Files changed:\n{diff_stat}\n\n"
            "A new commit will be created that undoes these changes.\n"
            "Call again with confirm=true to proceed.",
            binding,
        )
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception:
        status = ""
    if status:
        return _vcs_result(
            "⚠️ REVERT_ERROR: Working directory is not clean.\n"
            "Commit or discard changes first (use vcs_restore), then retry.",
            binding,
        )
    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "revert", "--no-edit", full_sha], cwd=repo_dir)
        except Exception as e:
            try:
                run_cmd(["git", "revert", "--abort"], cwd=repo_dir)
            except Exception:
                pass
            return _vcs_result(f"⚠️ REVERT_ERROR: git revert failed: {e}", binding)
    finally:
        _release_git_lock(lock)
    return _vcs_result(
        f"Reverted commit {full_sha[:8]}: {commit_msg}\nNew revert commit created.",
        binding,
    )
