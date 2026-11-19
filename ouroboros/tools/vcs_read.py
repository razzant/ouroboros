"""Read local Git history and identity from a dispatch-selected repository.

``repository`` selects a worktree before policy guards run; ``path`` is only a
literal filter inside it. These reads never fetch, stage, or serialize an index.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ouroboros.config import GIT_READ_CAPTURE_TIMEOUT_SEC
from ouroboros.protected_artifacts import block_reason_for_path, protected_artifact_paths
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.git_plumbing import _publish_git_error, _sanitize_git_error
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

READ_DESCRIPTION = (
    "Read local Git status, diff, history, files, blame, refs, ancestry, or repository identity. "
    "root/repository selects a worktree; path filters within it. No fetch. Diff uses configured "
    "filters: base compares to worktree (or index with staged); base+head compares two trees. "
    "Identity is sequential observations, not proof of an exact staged subject. "
    "History output has a physical capture bound; narrow with max_count, ref, or path."
)

_CONTENT_ACTIONS = frozenset({"show", "file", "blame"})
_ACTIONS = frozenset({"status", "diff", "log", "show", "file", "blame", "refs", "rev_parse", "merge_base", "identity"})
_PATH_ACTIONS = frozenset({"status", "diff", "log", "show", "file", "blame"})
_PAIR_ACTIONS = frozenset({"merge_base", "diff"})  # the two that compare revisions: base/head, never ref
_NO_REF_ACTIONS = frozenset({"merge_base", "identity", "status", "diff"})
_REVISION_ACTIONS = frozenset({"log", "show", "file", "blame", "rev_parse"})  # ref names their revision
_SUMMARY_ACTIONS = frozenset({"diff", "show"})  # stat / name_only summarize a comparison or a commit
_CAPTURE_BYTES = 1024 * 1024
_CONFIG_LOCATIONS = frozenset({"GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM"})


def _capture_read_git(repo: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Capture one local Git read with the inherited Git environment and helpers disabled.

    The user and system config stay read, from the same locations status and diff
    use: their filters, excludes, autocrlf and safe.directory decide what "clean" means.
    """
    from ouroboros.git_observation import host_git_env

    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_") or key in _CONFIG_LOCATIONS}
    env.update({
        "LC_ALL": "C", "LANG": "C", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
    })
    cmd = [
        "git", "--no-pager", "--no-optional-locks",
        "-c", "core.fsmonitor=false", "-c", "core.pager=cat", "-c", "color.ui=false",
        "-c", f"core.hooksPath={os.devnull}",
        "-c", "log.showSignature=false",
    ]
    from ouroboros import work_domain

    if work_domain.is_work(repo):  # its config, links and objects are the work domain's: read there, flags kept
        return _capture_work_git(repo, [*cmd, *args], host_git_env(env))

    # Drain both pipes concurrently, retaining only bounded prefixes. A large
    # object never first accumulates in communicate()/capture_output memory.
    def drain(stream, limit):
        prefix, total = bytearray(), 0
        while chunk := stream.read(65536):
            total += len(chunk)
            prefix.extend(chunk[:max(0, limit - len(prefix))])
        return bytes(prefix), total - len(prefix)

    with subprocess.Popen([*cmd, *args], cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=host_git_env(env)) as process, ThreadPoolExecutor(max_workers=2) as pool:
        stdout = pool.submit(drain, process.stdout, _CAPTURE_BYTES)
        stderr = pool.submit(drain, process.stderr, 65536)
        try:
            process.wait(timeout=GIT_READ_CAPTURE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        out, omitted = stdout.result()
        err, _ = stderr.result()
    completed = subprocess.CompletedProcess(cmd, process.returncode, out, err)
    completed.stdout_omitted = omitted
    return completed


def _capture_work_git(repo: pathlib.Path, cmd: list[str], env: dict) -> subprocess.CompletedProcess[bytes]:
    """``_capture_read_git`` for a work repository: the same command and flags run as the work domain
    (``work_domain.run``), streamed into service temporaries so only the same bounded prefixes are kept."""
    import tempfile

    from ouroboros import work_domain

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        done = work_domain.run(cmd, cwd=str(repo), env=env, stdout=out, stderr=err,
                               timeout=GIT_READ_CAPTURE_TIMEOUT_SEC)
        total = out.seek(0, os.SEEK_END)
        out.seek(0)
        err.seek(0)
        prefix = out.read(_CAPTURE_BYTES)
        completed = subprocess.CompletedProcess(cmd, done.returncode, prefix, err.read(65536))
    completed.stdout_omitted = total - len(prefix)
    return completed


def _git_failure(completed: subprocess.CompletedProcess[bytes]) -> RuntimeError:
    detail = completed.stderr.decode("utf-8", "replace").strip() or f"exit {completed.returncode}"
    return RuntimeError(detail)


def _run_read_git(repo: pathlib.Path, args: list[str], *, preserve_output: bool = False) -> str:
    completed = _capture_read_git(repo, args)
    if completed.returncode:
        raise _git_failure(completed)
    text = completed.stdout.decode("utf-8", "backslashreplace")
    text = text if preserve_output else text.rstrip()
    if omitted := getattr(completed, "stdout_omitted", 0):
        text += (f"\n⚠️ OUTPUT_TRUNCATED: captured first {_CAPTURE_BYTES} stdout bytes; "
                 f"{omitted} bytes omitted. Narrow with max_count (log), ref, or path.")
    return text


def _object(repo: pathlib.Path, ref: str, kind: str, *, field: str = "ref") -> str:
    if not ref.strip() or "\x00" in ref or "\n" in ref:
        raise ValueError(f"{field} must be a nonempty local Git revision")
    try:
        oid = _run_read_git(repo, ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{{kind}}}"])
    except RuntimeError as exc:
        # Which argument did not resolve and where to look, with Git's whole answer beside it: an absent,
        # ambiguous or other-typed object and an unsafe repository read alike from the outside.
        article = "an" if kind[:1] in "aeiou" else "a"
        raise RuntimeError(f"{field}={ref!r} did not resolve to {article} {kind} here (git: {exc}); nothing was read; "
                           "list revisions with action=refs or action=log") from exc
    if len(oid) not in (40, 64) or any(c not in "0123456789abcdef" for c in oid):
        raise ValueError("Git did not resolve one exact object")
    return oid


def repository_entry(ctx, root: str, repo: pathlib.Path, path: str, *, worktree_spelling: bool) -> str:
    """The repository-relative spelling both Git and the policy checks use for ``path``.

    An absolute path (a Docker workspace's backend spelling mapped to the host first) becomes
    relative with its folders resolved and its final entry kept: a symlink names itself, as in
    Git's tree. status/diff also take the retired tools' worktree spellings (a redundant
    root-basename prefix); history names an entry as given, since a folder absent today may
    exist in an older tree.
    """
    from ouroboros.shell_parse import is_absolute_path_text
    from ouroboros.tool_access_paths import normalize_root_relative
    from ouroboros.utils import normalized_path_text

    text = normalized_path_text(path)
    if not is_absolute_path_text(text):
        return normalize_root_relative(repo, text) if worktree_spelling else text
    if root == "active_workspace":
        from ouroboros.workspace_executor import executor_ref_from_ctx, map_backend_path_lexical

        executor = executor_ref_from_ctx(ctx)
        if executor is not None and executor.kind == "docker_exec":
            try:
                text = str(map_backend_path_lexical(executor, text))
            except ValueError:
                pass  # a host spelling, not a backend one
    entry = pathlib.Path(os.path.normpath(text))
    try:
        return (entry.parent.resolve(strict=False) / entry.name).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return text  # outside the repository: callers refuse it


def _literal_path(ctx, binding: ResolvedResourceBinding, path: str, *, worktree_spelling: bool = False) -> str:
    if not path:
        return ""
    # The binding proves current physical confinement; Git needs the requested
    # tree entry, not a symlink's current target (which may have different bytes).
    repo = binding.repository_path or binding.base_path
    if not binding.target_path.is_relative_to(repo):
        raise ValueError("path escapes the selected repository")
    relative = pathlib.PurePosixPath(
        repository_entry(ctx, binding.root, repo, path, worktree_spelling=worktree_spelling)).as_posix()
    if relative.startswith("/") or ".." in pathlib.PurePosixPath(relative).parts:
        raise ValueError("path must be repository-relative")
    return "" if relative in {"", "."} else relative


def _protected_content_reason(ctx, binding: ResolvedResourceBinding, path: str, action: str) -> str:
    operation = "read_bytes" if action == "file" else "static_introspection"
    if path:
        if reason := block_reason_for_path(ctx, binding.target_path, operation, binding):
            return reason
    # Historical entries may be gone from the worktree. Compare policy ancestry
    # as paths, without relying on Path.is_dir() for a removed protected folder.
    target = binding.target_path if path else (binding.repository_path or binding.base_path)
    for protected in protected_artifact_paths(ctx, binding):
        try:
            protected.relative_to(target)
        except ValueError:
            try:
                target.relative_to(protected)
            except ValueError:
                continue
        if reason := block_reason_for_path(ctx, protected, operation, binding):
            return reason
    return ""


def _identity(repo: pathlib.Path) -> str:
    toplevel = _run_read_git(repo, ["rev-parse", "--show-toplevel"])
    if pathlib.Path(toplevel).resolve(strict=False) != repo.resolve(strict=False):
        raise ValueError("repository must name the Git worktree root")
    git_dir = _run_read_git(repo, ["rev-parse", "--absolute-git-dir"])
    common_dir = _run_read_git(repo, ["rev-parse", "--git-common-dir"])
    common_path = pathlib.Path(common_dir)
    if not common_path.is_absolute():
        common_path = repo / common_path
    common_path = common_path.resolve(strict=False)
    branch_probe = _capture_read_git(repo, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch_probe.returncode == 0:
        branch = branch_probe.stdout.decode("utf-8", "replace").strip()
    elif branch_probe.returncode == 1 and not branch_probe.stdout and not branch_probe.stderr:
        branch = None  # Detached HEAD is the documented quiet exit.
    else:
        raise _git_failure(branch_probe)
    head_probe = _capture_read_git(repo, ["rev-parse", "--verify", "HEAD^{commit}"])
    if head_probe.returncode == 0:
        head = head_probe.stdout.decode("ascii", "replace").strip()
    else:
        # An unborn branch has a valid symbolic HEAD but no branch ref. A
        # broken existing ref, detached failure, or any other Git failure must
        # remain an error rather than masquerading as an empty repository.
        if not branch or head_probe.stdout:
            raise _git_failure(head_probe)
        full_ref = _run_read_git(repo, ["symbolic-ref", "HEAD"])
        missing = _capture_read_git(repo, ["show-ref", "--verify", "--quiet", full_ref])
        if (missing.returncode != 1 or missing.stdout or missing.stderr
                or (common_path / full_ref).exists()):
            raise _git_failure(head_probe)
        head = None
    return json.dumps({
        "worktree": toplevel,
        "git_dir": str(pathlib.Path(git_dir).resolve(strict=False)),
        "common_dir": str(common_path),
        "head": head,
        "branch": branch,
        "dirty": bool(_run_read_git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])),
    }, ensure_ascii=False, sort_keys=True)


def _argument_problems(action: str, *, path: str, ref: object, base: object, head: object, staged: bool, stat: bool,
                       name_only: bool, max_count: object, max_chars: object) -> list[str]:
    """Every constraint one call violates, each naming field, value and repair (DEVELOPMENT "LLM-first
    affordances"): one phrase for all of them was retried unchanged.
    Only what the call says about itself is judged here; Git's answers stay Git's."""
    problems = []
    if action not in _ACTIONS:
        return [f"action={action!r}: use one of {', '.join(sorted(_ACTIONS))}"]

    def given(value: object) -> bool:  # presence, not truthiness: False, 0.0, [] and {} were said, not omitted
        return value is not None and value != ""
    if not isinstance(max_count, int) or not 1 <= max_count <= 200:
        problems.append(f"max_count={max_count!r}: an integer from 1 to 200 (the number of commits log reads)")
    if not isinstance(max_chars, int) or max_chars < 0:
        problems.append(f"max_chars={max_chars!r}: 0 (no limit) or a positive integer")
    if path and action not in _PATH_ACTIONS:
        problems.append(f"path={path!r} applies to {', '.join(sorted(_PATH_ACTIONS))}; action={action} takes no path")
    if action in {"file", "blame"} and not path:
        problems.append(f"action={action} requires path (the file's repository-relative path)")
    if (given(base) or given(head)) and action not in _PAIR_ACTIONS:
        named = ", ".join(f"{name}={value!r}" for name, value in (("base", base), ("head", head)) if given(value))
        repair = (f"for action={action} name the revision with ref" if action in _REVISION_ACTIONS
                  else "action=refs filters with ref=refs/..." if action == "refs"
                  else f"action={action} takes no revision; omit base and head")
        problems.append(f"{named}: base and head apply to diff and merge_base; {repair}")
    if action == "merge_base" and not (given(base) and given(head)):
        problems.append("action=merge_base requires both base and head")
    if action == "diff" and given(head) and (not given(base) or staged):
        problems.append(f"head={head!r} compares two trees: diff needs base beside it and staged=false")
    for name, value, actions in (("staged", staged, {"diff"}), ("stat", stat, _SUMMARY_ACTIONS),
                                 ("name_only", name_only, _SUMMARY_ACTIONS)):
        if value and action not in actions:
            problems.append(f"{name}=true applies only to {' and '.join(sorted(actions))}; for action={action} omit {name}")
    if given(ref) and action in _NO_REF_ACTIONS:
        repair = f"{action} takes base and head" if action in _PAIR_ACTIONS else "omit ref"
        problems.append(f"ref={ref!r} does not apply to action={action}; {repair}")
    elif given(ref) and action == "refs" and (not isinstance(ref, str) or not ref.startswith("refs/")
                                       or any(c in ref for c in "\x00\n\r :*?[\\")):
        problems.append(f"ref={ref!r}: for action=refs the filter is a literal refs/ prefix, e.g. refs/heads/")
    taken = {"ref": action in _REVISION_ACTIONS, "base": action in _PAIR_ACTIONS, "head": action in _PAIR_ACTIONS}
    for name, value in (("ref", ref), ("base", base), ("head", head)):
        if not taken[name] or not given(value):
            continue
        if not isinstance(value, str):
            problems.append(f"{name}={value!r}: a local Git revision, written as text")
        elif not value.strip() or "\x00" in value or "\n" in value:
            problems.append(f"{name}={value!r}: a nonempty local Git revision")
    return problems


def _vcs_read(
    ctx,
    action: str,
    root: str = "active_workspace",
    repository: str = "",
    path: str = "",
    ref: str = "",
    base: str = "",
    head: str = "",
    staged: bool = False, stat: bool = False, name_only: bool = False,
    max_count: int = 20,
    max_chars: int = 0,
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    """Read Git objects using only the prepared repository and path binding."""
    max_count, max_chars = (int(value) if isinstance(value, str) and value.strip().isdigit() else value
                            for value in (max_count, max_chars))
    ref, base, head = (str(value) if isinstance(value, int) and not isinstance(value, bool) else value
                       for value in (ref, base, head))
    if problems := _argument_problems(action, path=path, ref=ref, base=base, head=head, staged=staged, stat=stat,
                                      name_only=name_only, max_count=max_count, max_chars=max_chars):
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (vcs_read): " + "; ".join(problems) + ". Nothing was read.",
            meta={"operation_outcome": "completed_no_effect"},
        ))
    try:
        if _resolved_binding is None:
            # Internal callers receive the same pre-dispatch preparation as API
            # callers; there is no handler-local repository retargeting.
            from ouroboros.tools.tool_resolution import _build_builtin_target_binding

            binding = _build_builtin_target_binding(ctx, "vcs_read", {
                "action": action, "root": root, "repository": repository, "path": path,
            })
        else:
            binding = _resolved_binding
        if binding.operation != "vcs" or binding.root not in {"active_workspace", "system_repo"}:
            raise ValueError("vcs_read requires a prepared VCS worktree binding")
        repo = binding.repository_path or binding.base_path
        if action in {"status", "diff"}:
            from ouroboros.tools.git_vcs_ops import _git_diff, _git_status

            common = dict(path=path, max_chars=max_chars, root=root, _resolved_binding=binding)
            return (_git_status(ctx, **common) if action == "status" else
                    _git_diff(ctx, staged=staged, stat=stat, name_only=name_only, base=base, head=head, **common))
        literal = _literal_path(ctx, binding, path)
        if action in {"file", "blame"} and not literal:
            raise ValueError(f"{action} requires path")
        if action in _CONTENT_ACTIONS:
            if reason := _protected_content_reason(ctx, binding, literal, action):
                return _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_POLICY_BLOCKED", text=reason))
        if action == "identity":
            output = _identity(repo)
        else:
            # Verify the selected worktree once before any history command.
            if pathlib.Path(_run_read_git(repo, ["rev-parse", "--show-toplevel"])).resolve(strict=False) != repo.resolve(strict=False):
                raise ValueError("repository must name the Git worktree root")
            if action == "log":
                cmd = ["log", "--no-color", f"--max-count={max_count}", "--format=fuller"]
                if ref:
                    cmd.append(_object(repo, ref, "commit"))
                if literal:
                    cmd.extend(["--", f":(literal){literal}"])
                output = _run_read_git(repo, cmd)
            elif action == "show":
                summary = ["--name-only"] if name_only else ["--stat"] if stat else []
                cmd = ["show", "--no-ext-diff", "--no-textconv", "--no-color", "--format=fuller", *summary,
                       _object(repo, ref or "HEAD", "commit")]
                if literal:
                    cmd.extend(["--", f":(literal){literal}"])
                output = _run_read_git(repo, cmd)
            elif action == "file":
                # cat-file names the object literally: a missing path is an error, never a pathspec match.
                output = _run_read_git(repo, ["cat-file", "blob", f"{_object(repo, ref or 'HEAD', 'commit')}:{literal}"], preserve_output=True)
            elif action == "blame":
                output = _run_read_git(repo, ["blame", "--no-ext-diff", "--no-textconv", _object(repo, ref or "HEAD", "commit"), "--", literal])
            elif action == "refs":  # the filter's shape was judged with the other arguments above
                output = _run_read_git(repo, ["for-each-ref", "--format=%(refname) %(objectname) %(objecttype)", *( [ref] if ref else [] )])
            elif action == "rev_parse":
                output = _object(repo, ref or "HEAD", "object")
            else:
                output = _run_read_git(repo, ["merge-base", _object(repo, base, "commit", field="base"),
                                              _object(repo, head, "commit", field="head")])
        from ouroboros.tools.git_vcs_ops import _limit_git_output, _vcs_result

        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="OK",
            text=_vcs_result(_limit_git_output(output, max_chars), binding, preserve_text=action == "file"),
        ))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return _publish_git_error(ctx, f"⚠️ GIT_ERROR: {_sanitize_git_error(str(exc))}")


def get_tools():
    """The history reader's public schema lives with its operation contract."""
    from ouroboros.tools.registry import ToolEntry

    return [ToolEntry("vcs_read", {
        "name": "vcs_read", "description": READ_DESCRIPTION,
        "parameters": {"type": "object", "required": ["action"], "properties": {
            "action": {"type": "string", "enum": sorted(_ACTIONS)},
            "root": {"type": "string", "enum": ["active_workspace", "system_repo"],
                     "default": "active_workspace",
                     "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
            "repository": {"type": "string", "default": "",
                           "description": "Git worktree root within root; omitted selects root itself. Not a file filter."},
            "path": {"type": "string", "default": "",
                     "description": "Repository-relative path filter (a pathspec for status/diff, literal for history); required for file and blame."},
            "ref": {"type": "string", "default": "",
                    "description": "Local revision (default HEAD), or a literal refs/ prefix for refs."},
            "base": {"type": "string", "default": "", "description": "First revision for merge_base, or diff base resolved once to a tree."},
            "head": {"type": "string", "default": "", "description": "Second revision for merge_base or diff; diff requires base and staged=false."},
            "staged": {"type": "boolean", "default": False, "description": "Diff the index (--staged); diff only."},
            "stat": {"type": "boolean", "default": False, "description": "Changed-file summary (--stat) of a diff or a shown commit."},
            "name_only": {"type": "boolean", "default": False, "description": "Changed file names only, for diff or show; takes precedence over stat."},
            "max_count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20,
                          "description": "Maximum commits returned by log."},
            "max_chars": {"type": "integer", "minimum": 0, "default": 0,
                          "description": "Rendered character limit; zero adds no limit beyond the history capture bound."},
        }},
    }, _vcs_read, is_code_tool=True, safety_policy="skip")]
