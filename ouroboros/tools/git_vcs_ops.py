"""Generic VCS inspection and rollback operations.

The ``ouroboros/tools/git.py`` facade re-exports these definitions. Shared
helpers are read through the call-time handle ``_git()`` so tests can patch the
facade binding. Displayed diffs use this leaf's bytes runner; parsed output
keeps ``run_cmd``. ``_sanitize_git_error`` and ``format_protected_paths`` bind
their owners at import time, including calls from f-strings.
"""

from __future__ import annotations

import json
import pathlib
import posixpath
from typing import List, Optional

from ouroboros.git_observation import host_git_env, numstat_paths
from ouroboros.runtime_mode_policy import format_protected_paths
from ouroboros.tools.registry import ToolContext
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.git_plumbing import _sanitize_git_error
from ouroboros.tools.git_plumbing import _publish_git_error
from ouroboros.tools.tool_result import ToolResult, LegacyTextResultAdapter, _publish_tool_result, _replace_tool_result


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import git

    return git


def _limit_git_output(text: str, max_chars: int = 0) -> str:
    limit = int(max_chars or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n⚠️ OUTPUT_TRUNCATED: git output limited to {limit} characters by max_chars. Narrow with path, or ref/max_count for history."


_NOT_UTF8_NOTE = (
    "\n⚠️ OUTPUT_NOT_UTF8: Git's output contained bytes that are not valid UTF-8, rendered as \\xNN "
    "escapes. Git can produce them itself, for example by cutting a combined diff's hunk-header context "
    "mid-character, so they do not by themselves mean a file is invalid UTF-8."
)


def _git_display_text(cmd: List[str], repo_dir: pathlib.Path) -> tuple[str, str]:
    """Run ``cmd`` as ``run_cmd`` does, but decode its bytes here; return (text, note).

    Git writes bytes, not one file's text: the combined diff of an unmerged path
    cuts its hunk-header context at a byte count, even through a character of a
    valid UTF-8 file. Strict UTF-8 comes first, so valid output reads as
    ``run_cmd`` returned it under a UTF-8 locale (text-mode newlines, stripped).
    Otherwise the bytes that are not UTF-8 become backslash escapes and the note
    discloses them, the policy of ``capture_staged_diff``; the caller appends the
    note after any ``max_chars`` cut. Environment, failure text and the absent
    timeout are ``run_cmd``'s.
    """

    def decode(raw: bytes) -> tuple[str, bool]:
        try:
            text, escaped = raw.decode("utf-8"), False
        except UnicodeDecodeError:
            text, escaped = raw.decode("utf-8", "backslashreplace"), True
        return text.replace("\r\n", "\n").replace("\r", "\n"), escaped

    from ouroboros import work_domain  # a work repository's filters run in its container, never on the service

    res = work_domain.run(cmd, cwd=str(repo_dir), capture_output=True, env=host_git_env())
    stdout, escaped = decode(res.stdout)
    if res.returncode != 0:
        stderr, stderr_escaped = decode(res.stderr)
        note = _NOT_UTF8_NOTE if escaped or stderr_escaped else ""
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}{note}"
        )
    return stdout.strip(), _NOT_UTF8_NOTE if escaped else ""


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


def _vcs_result(text: str, binding: ResolvedResourceBinding, *, preserve_text: bool = False) -> str:
    receipt = f"VCS target: root={binding.root}; repo={binding.repository_path or binding.base_path}"
    rendered = str(text or "") if preserve_text else str(text or "").rstrip()
    if preserve_text:
        return f"{rendered}\n\n{receipt}"
    return f"{rendered}\n\n{receipt}" if rendered else receipt


def _vcs_validation_refusal(ctx: ToolContext, text: str) -> str:
    """Only explicit argument checks before Git dispatch own no-effect proof."""
    result = LegacyTextResultAdapter.from_text("vcs", text)
    return _publish_tool_result(ctx, _replace_tool_result(
        result, meta_updates={"operation_outcome": "completed_no_effect"}))


def _binding_relative_path(ctx: ToolContext, binding: ResolvedResourceBinding, requested: str) -> str:
    from ouroboros.tools.vcs_read import _literal_path

    return _literal_path(ctx, binding, requested, worktree_spelling=True)


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
        if relative := _git()._binding_relative_path(ctx, binding, path):
            cmd.extend(["--", relative])  # an ordinary pathspec, as before: globs keep working
        output = _git().run_cmd(cmd, cwd=binding.repository_path or binding.base_path)
        return _git()._vcs_result(
            _git()._limit_git_output(output, max_chars),
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
    **kwargs,
) -> str:
    try:
        binding = _git()._vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        repo_dir = binding.repository_path or binding.base_path
        base, head = str(kwargs.get("base") or ""), str(kwargs.get("head") or "")
        # vcs_read refused head without base, or with staged, among the call's other argument problems.
        from ouroboros.tools.vcs_read import _object

        comparison = {}
        revisions = []
        for name, ref in (("base", base), ("head", head)):
            if ref:
                tree = _object(repo_dir, ref, "tree", field=name)  # an unresolved one is named with Git's answer
                revisions.append(tree)
                comparison.update({f"{name}_ref": ref, f"{name}_tree": tree})
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if name_only:
            # With index refresh disabled, name-only can mistake stat dirt for edits.
            cmd.extend(["--numstat", "-z"])
        elif stat:
            cmd.append("--stat")
        cmd.extend(revisions)
        if relative := _git()._binding_relative_path(ctx, binding, path):
            cmd.extend(["--", relative])
        if not (stat or name_only):
            from ouroboros.tools.vcs_read import _protected_content_reason

            # A pathspec pattern can reach protected files its literal target does not name.
            scope = "" if any(ch in relative for ch in "*?[") else relative
            if reason := _protected_content_reason(ctx, binding, scope, "show"):
                return _git()._vcs_result(reason, binding)
        from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason

        protected_block = protected_artifact_shell_block_reason(
            ctx, cmd, cwd=str(repo_dir), default_cwd=repo_dir, binding=binding,
        )
        if protected_block:
            return _git()._vcs_result(protected_block, binding)
        # name_only paths are parsed, not displayed: only displayed output takes escapes.
        output, note = (_git().run_cmd(cmd, cwd=repo_dir), "") if name_only else _git_display_text(cmd, repo_dir)
        if name_only:
            paths = numstat_paths(output, cmd, lambda args: _git().run_cmd(args, cwd=repo_dir))
            output = "\n".join(json.dumps(path, ensure_ascii=False)
                               if any(ord(c) < 32 or c in '\\"' for c in path) else path for path in paths)
        text = _git()._vcs_result(_git()._limit_git_output(output, max_chars) + note, binding)
        if comparison:
            comparison["kind"] = "tree_to_tree" if head else "tree_to_index" if staged else "tree_to_worktree"
            text += "\nComparison: " + json.dumps(comparison, ensure_ascii=False, sort_keys=True)
            return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta={"comparison": comparison}))
        return text
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
        return ("⚠️ PULL_ERROR: Not on a named branch (detached HEAD). Cannot pull. "
                "Inspect HEAD and local branches first. Use `git switch -c <new-branch> HEAD` "
                "with an unused name, or `git switch <existing-branch>` to select an existing branch; "
                "neither force-resets it.")
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
            return _vcs_validation_refusal(ctx, _git()._vcs_result(
                f"⚠️ RESTORE_ERROR: pathspec magic is not supported: {_raw}", binding,
            ))
        _collapsed = posixpath.normpath(_git().normalize_repo_path(_raw))
        if posixpath.isabs(_collapsed) or _collapsed == ".." or _collapsed.startswith("../"):
            return _vcs_validation_refusal(ctx, _git()._vcs_result(
                f"⚠️ RESTORE_ERROR: path escapes the repository root: {_raw}", binding,
            ))
        normalized_paths.append(_collapsed)
    if paths and not normalized_paths:
        return _vcs_validation_refusal(ctx, _git()._vcs_result("⚠️ RESTORE_ERROR: No valid paths provided.", binding))
    repo_dir = binding.base_path
    # The paths, else this whole folder: never the rest of a repository that encloses it.
    scope = normalized_paths or ["."]
    try:
        # One NUL-delimited porcelain read of the scope is judged, previewed, acted on and re-read
        # alike. run_cmd() strips its stdout, and a worktree-only modification renders as " M path",
        # so the stripped first line once lost its leading space and the column parser dropped the
        # first CHARACTER of that path — the protected gate then judged "IBLE.md" while the restore
        # acted on BIBLE.md.
        changes = _scope_changes(pathlib.Path(repo_dir), scope)
    except Exception as e:
        return _git()._vcs_result(f"⚠️ RESTORE_ERROR: git status failed: {e}", binding)
    dirty_files = [path for _state, path in changes]
    if not dirty_files:
        return _git()._vcs_result("Nothing to restore — no uncommitted change matches paths." if normalized_paths
                                  else "Nothing to restore — working directory is already clean.", binding)
    targets_system = _git().binding_targets_system_repo(ctx, binding)
    affected_protected = _git().protected_paths_in(dirty_files) if targets_system else []
    if normalized_paths and targets_system:
        # A pathspec is not a path: git expands directories, fnmatch wildcards,
        # and "." to a FILE SET, so judging the requested string alone let
        # `paths=["."]`, `["ouroboros"]`, or `["ouroboros/safety.*"]` reach dirty
        # protected files past a gate that only knew exact names. Judge the
        # damage set instead: the files this restore would actually mutate, resolved
        # by git itself from the requested pathspecs; the action below acts on exactly them.
        for norm in normalized_paths:
            if _git().is_protected_runtime_path(norm):
                return _git()._vcs_result(
                    f"⚠️ RESTORE_BLOCKED: Cannot restore protected file: {norm}. "
                    "Protected core/contract/release paths must be changed through reviewed commits.",
                    binding,
                )
        # The scoped porcelain, not `git ls-files`, which resolves against the index and so
        # cannot see a STAGED deletion or rename-away of a protected file (the path is gone
        # from the index but a restore would resurrect it, discarding the protected staged
        # change). Porcelain lists staged deletes, rename sources, and untracked files alike.
        damage_protected = sorted({
            f for f in dirty_files if f and _git().is_protected_runtime_path(f)
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
        try:  # HEAD to working files: staged and unstaged edits, additions and deletions alike
            diff_stat = _git().run_cmd(["git", "diff", "--stat", "HEAD", "--", *scope], cwd=repo_dir).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        preview = ["Uncommitted changes that will be lost:", "", diff_stat]
        for title, names in (("New files in the index that will be deleted:", [p for s, p in changes if s[0] == "A"]),
                             ("Untracked files that will be removed:", [p for s, p in changes if s == "??"])):
            if names:
                preview += ["", title, *(f"  {_shown(f)}" for f in names[:15])]
                preview += [f"  … and {len(names) - 15} more"] if len(names) > 15 else []
        preview.append("")
        preview.append("Call again with confirm=true to proceed.")
        return _git()._vcs_result("\n".join(preview), binding)
    from ouroboros.git_observation import _pathspec_batches

    tracked = [path for state, path in changes if state != "??"]
    try:  # status names rows from the repository's top level, which may enclose this folder: act there
        top = pathlib.Path(repo_dir) / _git().run_cmd(["git", "rev-parse", "--show-cdup"], cwd=repo_dir)
        ancestors = {str(parent) for path in tracked for parent in pathlib.PurePosixPath(path).parents
                     if str(parent) != "."}
        parent_rows = _scope_changes(top, [f":(literal){name}" for name in sorted(ancestors)]) if ancestors else []
        outside_index = {path for state, path in parent_rows if path in ancestors and path not in tracked
                         and state[0] not in {" ", "D", "?"}}
    except Exception as e:
        return _git()._vcs_result(f"⚠️ RESTORE_ERROR: repository or index paths could not be read: {e}", binding)
    # No overlay: an entry HEAD lacks (a staged addition, a rename's new side) leaves the index
    # and the files too. Exact literal names of the read rows, in argv-sized batches: a request
    # naming only untracked files is unknown to checkout, and the action stays the judged set.
    checkout, restored = ["git", "checkout", "--no-overlay", "HEAD", "--"], 0
    added = {path for state, path in changes if state[0] == "A"}  # HEAD lacks them: checkout removes them
    # A staged parent file can have no working file at all. Writing a child would silently drop
    # that outside-scope index entry; filesystem occupancy alone cannot protect its saved bytes.
    index_held = {path for path in tracked
                  if any(str(parent) in outside_index for parent in pathlib.PurePosixPath(path).parents)}
    # Git writes a HEAD file by deleting whatever holds its path (a file-to-folder change, or a file where
    # its folder belongs), so such a file waits until the removals and the cleanup below cleared the way.
    later = [path for path in tracked if path in index_held or (path not in added
             and (_occupied(top, path) or any(name.startswith(path + "/") for name in added)))]
    try:
        for batch in _pathspec_batches(checkout, [path for path in tracked if path not in later]):
            _git().run_cmd([*checkout, *batch], cwd=top)
            restored += len(batch)
    except Exception as e:
        return _git()._vcs_result(
            f"⚠️ RESTORE_PARTIAL: restored {restored} of {len(tracked)} tracked path(s), then git checkout failed: {e}"
            if restored else f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
    # Only the original untracked names, never a new scan. -x keeps checkout's changed ignore
    # rules from hiding those targets; initially ignored files are absent from this exact set.
    clean, untracked = ["git", "clean", "-fdx", "--"], [path for state, path in changes if state == "??"]
    try:
        for batch in _pathspec_batches(clean, untracked):
            _git().run_cmd([*clean, *batch], cwd=top)
    except Exception as e:
        # Do not swallow: the checkout landed but untracked cleanup did not, so the
        # working tree does NOT fully match HEAD; a swallowed failure once claimed it did.
        return _git()._vcs_result(
            f"⚠️ RESTORE_PARTIAL: checked out {restored} path(s), but `git clean` failed and "
            f"untracked files may remain: {e}", binding)
    _prune_emptied(top, untracked, pathlib.Path(repo_dir))
    kept = [path for path in later if path in index_held or _occupied(top, path)]
    try:
        for batch in _pathspec_batches(checkout, [path for path in later if path not in kept]):
            _git().run_cmd([*checkout, *batch], cwd=top)
            restored += len(batch)
    except Exception as e:
        return _git()._vcs_result(
            f"⚠️ RESTORE_PARTIAL: restored {restored} of {len(tracked)} tracked path(s) and removed the untracked "
            f"ones, then git checkout failed: {e}" if restored or untracked
            else f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
    try:  # the claim below is observed, not assumed
        remaining = [_shown(path) for _state, path in _scope_changes(pathlib.Path(repo_dir), scope)]
    except Exception as e:
        remaining = [f"(the scope could not be re-read: {e})"]
    if remaining:
        return _git()._vcs_result(
            f"⚠️ RESTORE_PARTIAL: checkout and clean ran, but {len(remaining)} path(s) in scope still differ "
            f"from HEAD: {', '.join(remaining[:15])}" + (f" and {len(remaining) - 15} more" if len(remaining) > 15 else "")
            + (f". Not restored, to keep files or index entries that hold their path: {', '.join(map(_shown, kept))}" if kept else ""),
            binding)
    if normalized_paths:
        return _git()._vcs_result(f"Restored {len(dirty_files)} path(s) to HEAD.", binding)
    return _git()._vcs_result("All uncommitted changes discarded. Working directory matches HEAD.", binding)


def _scope_changes(repo_dir: pathlib.Path, paths: List[str]) -> List[tuple]:
    """The ``git status`` rows of a restore scope as (XY, path): staged and working changes with
    rename sides apart, and each untracked file; ignored files are never listed. ``paths`` resolve
    from ``repo_dir``, while each row names its path from the repository's top level."""
    from ouroboros import work_domain
    from ouroboros.git_observation import host_git_env

    # --untracked-files=all: a repo-local status.showUntrackedFiles=no would EMPTY the
    # untracked half, and dir-collapsing ("dir/") hides the files a protected judgment needs.
    res = work_domain.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
                           "--", *paths], cwd=repo_dir, capture_output=True, timeout=10, env=host_git_env())
    if res.returncode != 0:
        err = (res.stderr or b"").decode("utf-8", errors="replace").strip()[:200]
        raise RuntimeError(f"git status --porcelain=v1 -z failed (exit {res.returncode}): {err}")
    return [(row[:2].decode("ascii", "replace"), row[3:].decode("utf-8", "surrogateescape"))
            for row in res.stdout.split(b"\0") if len(row) > 3]


def _occupied(top: pathlib.Path, path: str) -> bool:
    """Whether writing HEAD's file at ``path`` would make Git delete something there: a folder that holds
    anything (ignored files included) at the path, or a file or link where one of its folders belongs."""
    from ouroboros import work_domain

    target = top / path
    try:
        if work_domain.is_work(target):  # its own metadata through the confined descent, never followed
            if work_domain.kind(target) == "dir":
                return bool(work_domain.entries(target))
            return any(work_domain.kind(top / folder) in ("link", "file")
                       for folder in pathlib.PurePosixPath(path).parents if str(folder) != ".")
        if target.is_dir() and not target.is_symlink():
            return next(target.iterdir(), None) is not None
        return any((top / folder).is_symlink() or (top / folder).is_file()
                   for folder in pathlib.PurePosixPath(path).parents if str(folder) != ".")
    except OSError:  # unreadable: never write over it
        return True


def _prune_emptied(top: pathlib.Path, names: List[str], folder: pathlib.Path) -> None:
    """Remove the folders that removing ``names`` left empty, inside ``folder`` only: rmdir deletes
    no content, so a folder still holding anything, an ignored file included, stays."""
    from ouroboros import work_domain  # a work folder is settled by its spelling and removed as the work domain

    work = work_domain.is_work(folder)
    root = work_domain.settle(folder)
    for name in names:
        parent = work_domain.settle((top / name).parent)
        while parent != root and root in parent.parents:
            try:
                work_domain.remove_empty_dir(parent) if work else parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _shown(path: str) -> str:
    """A path for display: bytes that are not UTF-8 become backslash escapes."""
    return path.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


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
        return _vcs_validation_refusal(ctx, _git()._vcs_result("⚠️ REVERT_ERROR: sha parameter is required.", binding))
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
    from ouroboros.consciousness_authority import effective_runtime_mode
    from ouroboros.runtime_mode_policy import mode_allows_protected_write

    mode = effective_runtime_mode(_git()._current_runtime_mode(), getattr(ctx, "task_metadata", None))
    if protected_changes and not (mode_allows_protected_write(mode) or _git()._authorized_managed_update_resolver(ctx)):
        return _git()._vcs_result(
            f"⚠️ CORE_PROTECTION_BLOCKED: Runtime mode {mode} cannot revert protected file(s): "
            f"{format_protected_paths(protected_changes)}. "
            "Mechanical revert is review-exempt, but retains the caller's runtime-mode authority.",
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
    lock = _git()._acquire_git_lock(ctx, repo_dir=repo_dir)  # the tree this revert writes, never the serving one
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
