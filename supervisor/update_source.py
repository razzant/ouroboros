"""Official managed-update source selection and bounded git network calls."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
from typing import Callable, List, Optional, Tuple


FETCH_TIMEOUT_RC = 124
MANAGED_TAG_NAMESPACE = "refs/ouroboros-managed/tags"
_RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def official_ref_has_constitution(
    ref: str, *, repo_dir: os.PathLike[str] | str | None = None
) -> bool:
    """Whether *ref* contains the non-empty regular ``BIBLE.md`` required by P4."""
    if repo_dir is None:
        from supervisor import git_ops as _g

        repo_dir = _g.REPO_DIR
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-l", ref, "--", "BIBLE.md"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    fields = (result.stdout or "").strip().split(maxsplit=4)
    return bool(
        result.returncode == 0
        and len(fields) == 5
        and fields[0] in {"100644", "100755"}
        and fields[1] == "blob"
        and fields[3].isdigit()
        and int(fields[3]) > 0
        and fields[4] == "BIBLE.md"
    )


def _git_network_bounded(
    args: List[str],
    *,
    timeout: Optional[float] = None,
    cwd: os.PathLike[str] | str | None = None,
) -> Tuple[int, str, str]:
    """Run one non-interactive git network command under the shared ceiling.

    ``cwd`` selects the repository the command runs in; the default remains the
    managed system repository. An explicit ``cwd`` must be an existing
    directory — network commands against a caller-selected repository must
    never silently fall back to the system repository.
    """
    from ouroboros.update_channels import get_managed_update_fetch_timeout_sec
    from supervisor import git_ops as _g

    if cwd is None:
        repo_dir = _g.REPO_DIR
    else:
        repo_dir = pathlib.Path(cwd)
        if not repo_dir.is_dir():
            return 127, "", f"git cwd is not a directory: {repo_dir}"
    limit = float(timeout or get_managed_update_fetch_timeout_sec())
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "GIT_TERMINAL_PROMPT": "0"}
    rc, out, err = _g._run_git_process_bounded(
        ["git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30", *args],
        timeout=limit,
        cwd=repo_dir,
        env=env,
        text=True,
    )
    if rc == FETCH_TIMEOUT_RC:
        return rc, "", f"git {' '.join(args[:2])} exceeded {limit:.0f}s and was terminated"
    return rc, str(out or "").strip(), str(err or "").strip()


def git_fetch_bounded(remote_name: str, *, timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """Fetch the official remote under one cross-platform wall-clock ceiling."""
    return _git_network_bounded(
        [
            "fetch",
            "--quiet",
            remote_name,
            f"+refs/heads/*:refs/remotes/{remote_name}/*",
            f"+refs/tags/*:{MANAGED_TAG_NAMESPACE}/*",
        ],
        timeout=timeout,
    )


def _managed_update_target(update_channel: str | None = None) -> Tuple[str, str, str]:
    """Return the runtime-selected official ref, independent from launcher provenance."""
    from supervisor import git_ops as _g

    managed_meta = _g._read_managed_repo_meta()
    if not managed_meta:
        return "", "", ""
    from ouroboros.update_channels import UPDATE_CHANNEL_BRANCHES, get_update_branch

    remote_name = _g._managed_remote_name(managed_meta)
    remote_branch = get_update_branch() if update_channel is None else UPDATE_CHANNEL_BRANCHES[update_channel]
    target_ref = f"{remote_name}/{remote_branch}" if remote_name and remote_branch else ""
    return remote_name, remote_branch, target_ref


def resolve_managed_update_target(
    remote_name: str,
    remote_branch: str,
    branch_ref: str,
    *,
    update_channel: str,
    capture: Callable[[List[str]], Tuple[int, str, str]],
) -> Tuple[str, str, str]:
    """Resolve the selected feed to one immutable commit.

    QA and Development follow their branch tips. Stable follows the newest plain
    ``vX.Y.Z`` release whose commit is reachable from both ``main`` and
    ``ouroboros-stable``. This keeps an untagged main commit out of the stable
    feed without creating a second channel-specific updater.
    """
    if not remote_name or not remote_branch or not branch_ref:
        return "", "", "managed update target is unavailable"
    if str(update_channel or "") != "stable":
        rc, sha, error = capture(
            ["git", "rev-parse", "--verify", f"{branch_ref}^{{commit}}"]
        )
        return (branch_ref, sha, "") if rc == 0 and sha else ("", "", error or branch_ref)

    main_ref = f"{remote_name}/main"
    qa_ref = f"{remote_name}/ouroboros-stable"
    for required_ref in (main_ref, qa_ref):
        rc, _sha, error = capture(
            ["git", "rev-parse", "--verify", f"{required_ref}^{{commit}}"]
        )
        if rc != 0:
            return "", "", error or f"stable release branch is unavailable: {required_ref}"

    rc, raw_tags, error = capture(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:strip=3)",
            f"{MANAGED_TAG_NAMESPACE}/v*",
        ]
    )
    if rc != 0:
        return "", "", error or "could not list release tags"
    candidates = []
    for tag in raw_tags.splitlines():
        match = _RELEASE_TAG_RE.fullmatch(tag.strip())
        if match:
            candidates.append((tuple(int(part) for part in match.groups()), tag.strip()))
    candidates.sort(reverse=True)

    for _version, tag in candidates:
        tag_ref = f"{MANAGED_TAG_NAMESPACE}/{tag}"
        rc, sha, _error = capture(
            ["git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"]
        )
        if rc != 0 or not sha:
            continue
        present_in_both = True
        for release_branch_ref in (main_ref, qa_ref):
            ancestor_rc, _out, _ancestor_error = capture(
                ["git", "merge-base", "--is-ancestor", sha, release_branch_ref]
            )
            if ancestor_rc != 0:
                present_in_both = False
                break
        if present_in_both:
            return tag_ref, sha, ""
    return "", "", "no shared stable vX.Y.Z release exists on main and ouroboros-stable"
