"""Generic VCS inspection and rollback operations, split out of
``ouroboros/tools/git.py`` (v7 module-size discipline). Every span is
extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py; the parent re-exports every moved name.
Parent-scope helpers the monolith read as module globals are read through
the call-time handle ``_git()`` — never a from-import — so the facade
binding stays the one tests monkeypatch. ``_sanitize_git_error`` and
``format_protected_paths`` are the f-string-read exceptions (the byte gate
cannot rewrite f-string internals): they bind their owners at import time.
"""

from __future__ import annotations

import pathlib
import posixpath
from typing import List, Optional

from ouroboros.runtime_mode_policy import format_protected_paths
from ouroboros.tools.registry import ToolContext
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.git_plumbing import _sanitize_git_error
from ouroboros.tools.git_plumbing import _publish_git_error


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import git

    return git


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

    resolved = binding or _git().build_resolved_resource_binding(
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
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        cmd = ["git", "status", "--porcelain"]
        if relative := _git()._binding_relative_path(binding, path):
            cmd.extend(["--", _git().safe_relpath(relative)])
        return _git()._vcs_result(
            _git()._limit_git_output(_git().run_cmd(cmd, cwd=binding.base_path), max_chars),
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
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        repo_dir = binding.base_path
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if name_only:
            cmd.append("--name-only")
        elif stat:
            cmd.append("--stat")
        if relative := _git()._binding_relative_path(binding, path):
            cmd.extend(["--", _git().safe_relpath(relative)])
        from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason

        protected_block = protected_artifact_shell_block_reason(
            ctx, cmd, cwd=str(repo_dir), default_cwd=repo_dir, binding=binding,
        )
        if protected_block:
            return _git()._vcs_result(protected_block, binding)
        return _git()._vcs_result(_git()._limit_git_output(_git().run_cmd(cmd, cwd=repo_dir), max_chars), binding)
    except Exception as e:
        return _publish_git_error(
            ctx,
            f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}",
        )


def _ff_pull(repo_dir: pathlib.Path) -> str:
    try:
        branch = _git().run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not determine current branch: {e}"
    if not branch or branch == "HEAD":
        return "⚠️ PULL_ERROR: Not on a named branch (detached HEAD). Cannot pull."
    try:
        _git()._run_git_network_cmd(["git", "fetch", "origin"], cwd=repo_dir)
    except Exception as e:
        return f"⚠️ PULL_ERROR: git fetch failed: {_sanitize_git_error(str(e))}"
    try:
        before_sha = _git().run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
        remote_sha = _git().run_cmd(
            ["git", "rev-parse", f"origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not resolve SHAs: {e}"
    if before_sha == remote_sha:
        return f"Already up to date. HEAD={before_sha[:8]} matches origin/{branch}."
    try:
        new_commits = _git().run_cmd(
            ["git", "log", "--oneline", f"HEAD..origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception:
        new_commits = "(could not list commits)"
    try:
        _git().run_cmd(["git", "merge", "--ff-only", f"origin/{branch}"], cwd=repo_dir)
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
        after_sha = _git().run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
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
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root)
        return _git()._vcs_result(_git()._ff_pull(binding.base_path), binding)
    except Exception as e:
        return f"⚠️ PULL_ERROR: {_sanitize_git_error(str(e))}"


def _restore_to_head(ctx: ToolContext, confirm: bool = False,
                     paths: Optional[List[str]] = None,
                     root: str = "system_repo",
                     _resolved_binding: Optional[ResolvedResourceBinding] = None) -> str:
    try:
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ RESTORE_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    try:
        # NUL-delimited porcelain via the shared review helper: run_cmd() strips
        # its stdout, and a worktree-only modification renders as " M path", so
        # the stripped first line lost its leading space and the column-based
        # line parser dropped the first CHARACTER of that path — the protected
        # gate then judged "IBLE.md" while the restore acted on BIBLE.md.
        from ouroboros.tools.review_helpers import list_changed_paths_from_git_status

        dirty_files = list_changed_paths_from_git_status(
            pathlib.Path(repo_dir), include_sources_for_renames=True,
        )
    except Exception as e:
        return _git()._vcs_result(f"⚠️ RESTORE_ERROR: git status failed: {e}", binding)
    if not dirty_files:
        return _git()._vcs_result("Nothing to restore — working directory is already clean.", binding)
    targets_system = _git().binding_targets_system_repo(ctx, binding)
    affected_protected = _git().protected_paths_in(dirty_files) if targets_system else []
    # Resolve each requested path to the SETTLED form git itself will act on, and
    # let both the protected-path gate below and the checkout/clean action consume
    # that identical value. A divergent normalizer was a protected-path bypass:
    # the gate used normalize_repo_path() while the action used lstrip("./"),
    # which strips leading '.' AND '/' characters, so `../ouroboros/safety.py`
    # was judged as a different (unprotected) string and then checked out against
    # the real protected file. Collapsing is the load-bearing half: git resolves
    # `..` inside a pathspec, while normalize_repo_path() does not, so a gate fed
    # the uncollapsed string reads `tests/../ouroboros/safety.py` as unprotected
    # and the action still lands on `ouroboros/safety.py`. Judge and act on the
    # collapsed value, and refuse anything that leaves the repository root.
    normalized_paths: List[str] = []
    for _p in (paths or []):
        _raw = str(_p or "").strip()
        if not _raw:
            continue
        if _raw.startswith(":"):
            # Pathspec magic (":/", ":(glob)", ":!", ...) re-scopes or negates a
            # pathspec behind the gate's back; a plain repo-relative path never
            # needs it, so refuse rather than judge a string git reads differently.
            return _git()._vcs_result(
                f"⚠️ RESTORE_ERROR: pathspec magic is not supported: {_raw}", binding,
            )
        _collapsed = posixpath.normpath(_git().normalize_repo_path(_raw))
        if posixpath.isabs(_collapsed) or _collapsed == ".." or _collapsed.startswith("../"):
            return _git()._vcs_result(
                f"⚠️ RESTORE_ERROR: path escapes the repository root: {_raw}", binding,
            )
        normalized_paths.append(_collapsed)
    if paths and not normalized_paths:
        return _git()._vcs_result("⚠️ RESTORE_ERROR: No valid paths provided.", binding)
    if normalized_paths and targets_system:
        # A pathspec is not a path: git expands directories, fnmatch wildcards,
        # and "." to a FILE SET, so judging the requested string alone let
        # `paths=["."]`, `["ouroboros"]`, or `["ouroboros/safety.*"]` reach dirty
        # protected files past a gate that only knew exact names. Judge the
        # damage set instead: the files this restore would actually mutate —
        # dirty tracked matches (checkout reverts them) and untracked matches
        # (`git clean -fd` deletes them) — resolved by git itself from the SAME
        # pathspecs the action below consumes.
        for norm in normalized_paths:
            if _git().is_protected_runtime_path(norm):
                return _git()._vcs_result(
                    f"⚠️ RESTORE_BLOCKED: Cannot restore protected file: {norm}. "
                    "Protected core/contract/release paths must be changed through reviewed commits.",
                    binding,
                )
        try:
            # The SAME pathspec-scoped porcelain the dirty set came from — not
            # `git ls-files`, which resolves against the index and so cannot see
            # a STAGED deletion or rename-away of a protected file (the path is
            # gone from the index but `checkout HEAD -- .` would resurrect it,
            # discarding the protected staged change). Porcelain lists staged
            # deletes, rename sources, and untracked files alike.
            from ouroboros.tools.review_helpers import list_changed_paths_from_git_status

            scoped_dirty = list_changed_paths_from_git_status(
                pathlib.Path(repo_dir), normalized_paths,
                include_sources_for_renames=True,
            )
        except Exception as e:
            return _git()._vcs_result(
                f"⚠️ RESTORE_ERROR: could not resolve pathspec matches: {e}", binding,
            )
        damage_protected = sorted({
            f for f in scoped_dirty if f and _git().is_protected_runtime_path(f)
        })
        if damage_protected:
            return _git()._vcs_result(
                f"⚠️ RESTORE_BLOCKED: pathspec matches protected file(s) with uncommitted "
                f"changes: {format_protected_paths(damage_protected)}. "
                "Protected core/contract/release paths must be changed through reviewed commits.",
                binding,
            )
    elif affected_protected:
        return _git()._vcs_result(
            f"⚠️ RESTORE_BLOCKED: Uncommitted changes touch protected file(s): "
            f"{format_protected_paths(affected_protected)}. "
            f"Use paths= to restore specific non-critical files, or resolve manually.",
            binding,
        )
    if not confirm:
        try:
            diff_stat = _git().run_cmd(["git", "diff", "--stat"], cwd=repo_dir).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        try:
            untracked = _git().run_cmd(
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
        return _git()._vcs_result("\n".join(preview), binding)
    if normalized_paths:
        # Reuse the exact contained, normalized paths the gate above judged.
        try:
            _git().run_cmd(["git", "checkout", "HEAD", "--"] + normalized_paths, cwd=repo_dir)
        except Exception as e:
            return _git()._vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            _git().run_cmd(["git", "clean", "-fd", "--"] + normalized_paths, cwd=repo_dir)
        except Exception as e:
            # Do not swallow: the checkout landed but untracked cleanup did not,
            # so the working tree does NOT fully match HEAD. Report it.
            return _git()._vcs_result(
                f"⚠️ RESTORE_PARTIAL: checked out {len(normalized_paths)} path(s), "
                f"but `git clean` failed and untracked files may remain: {e}",
                binding,
            )
        return _git()._vcs_result(f"Restored {len(normalized_paths)} path(s) to HEAD.", binding)
    else:
        try:
            _git().run_cmd(["git", "checkout", "HEAD", "--", "."], cwd=repo_dir)
        except Exception as e:
            return _git()._vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            _git().run_cmd(["git", "clean", "-fd"], cwd=repo_dir)
        except Exception as e:
            # A swallowed failure previously let this claim "matches HEAD" while
            # untracked files survived. Surface it instead of asserting a state
            # that was not verified.
            return _git()._vcs_result(
                f"⚠️ RESTORE_PARTIAL: tracked changes discarded, but `git clean` "
                f"failed and untracked files may remain: {e}",
                binding,
            )
        return _git()._vcs_result(
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
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ REVERT_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    sha = sha.strip()
    if not sha:
        return _git()._vcs_result("⚠️ REVERT_ERROR: sha parameter is required.", binding)
    try:
        full_sha = _git().run_cmd(
            ["git", "rev-parse", "--verify", sha], cwd=repo_dir,
        ).strip()
    except Exception:
        return _git()._vcs_result(f"⚠️ REVERT_ERROR: Commit '{sha}' not found.", binding)
    try:
        parents = _git().run_cmd(
            ["git", "rev-list", "--parents", "-1", full_sha], cwd=repo_dir,
        ).strip().split()
    except Exception:
        parents = [full_sha]
    if len(parents) > 2:
        return _git()._vcs_result(
            f"⚠️ REVERT_ERROR: Commit {sha[:8]} is a merge commit ({len(parents)-1} parents). "
            "git revert on merge commits requires specifying a parent.",
            binding,
        )
    try:
        changed_files = _git().run_cmd(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", full_sha],
            cwd=repo_dir,
        ).strip().splitlines()
    except Exception:
        changed_files = []
    protected_changes = (
        _git().protected_paths_in(changed_files)
        if _git().binding_targets_system_repo(ctx, binding)
        else []
    )
    if protected_changes:
        return _git()._vcs_result(
            f"⚠️ REVERT_BLOCKED: Commit {sha[:8]} touches protected file(s): "
            f"{format_protected_paths(protected_changes)}. "
            "Direct vcs_revert cannot create protected-path commits; stage the intended "
            "revert manually and use commit_reviewed so the normal triad + scope review covers it.",
            binding,
        )
    try:
        commit_msg = _git().run_cmd(
            ["git", "log", "-1", "--format=%s", full_sha], cwd=repo_dir,
        ).strip()
    except Exception:
        commit_msg = "(unknown)"
    if not confirm:
        try:
            diff_stat = _git().run_cmd(
                ["git", "diff", f"{full_sha}^..{full_sha}", "--stat"], cwd=repo_dir,
            ).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        return _git()._vcs_result(
            f"This will revert commit {full_sha[:8]}:\n"
            f"  Message: {commit_msg}\n"
            f"  Files changed:\n{diff_stat}\n\n"
            "A new commit will be created that undoes these changes.\n"
            "Call again with confirm=true to proceed.",
            binding,
        )
    try:
        status = _git().run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception:
        status = ""
    if status:
        return _git()._vcs_result(
            "⚠️ REVERT_ERROR: Working directory is not clean.\n"
            "Commit or discard changes first (use vcs_restore), then retry.",
            binding,
        )
    lock = _git()._acquire_git_lock(ctx)
    try:
        try:
            _git().run_cmd(["git", "revert", "--no-edit", full_sha], cwd=repo_dir)
        except Exception as e:
            try:
                _git().run_cmd(["git", "revert", "--abort"], cwd=repo_dir)
            except Exception:
                pass
            return _git()._vcs_result(f"⚠️ REVERT_ERROR: git revert failed: {e}", binding)
    finally:
        _git()._release_git_lock(lock)
    return _git()._vcs_result(
        f"Reverted commit {full_sha[:8]}: {commit_msg}\nNew revert commit created.",
        binding,
    )
