"""Durable, standalone projects for a genesis acting child: not a checkout, not in the private-copy
records and never pruned by age. ``subagent_worktrees`` still answers to these two names, lazily."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ouroboros import subagent_worktrees as sw


def provision_genesis_project(
    *,
    repo_dir: Any,
    task_id: Any,
    parent_task_id: str = "",
    projects_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    dir_name: str = "",
) -> sw.WorktreeHandle:
    """Provision a durable, isolated, EMPTY git project for a genesis acting child.

    Unlike a worktree this is a standalone repo (not a checkout of the live body)
    under the durable projects root. It is the deliverable itself and is NEVER
    GC-pruned, so it is intentionally not added to the worktree registry. The
    child builds the whole project here and returns a ``workspace.patch`` that is
    a diff from the empty initial commit (``base_sha``).

    ``dir_name`` names the genesis directory meaningfully (e.g. the project name,
    readable Unicode via ``project_folder_basename``) instead of the raw task id, so
    sibling builders share a recognizable project root; the handle's binding
    identity stays ``task_id`` (I, v6.39). Existing folders are never renamed.
    """
    from ouroboros.config import get_subagent_projects_root
    from ouroboros.project_facts import project_folder_basename

    repo_dir = Path(repo_dir).resolve()
    root = Path(projects_root) if projects_root else Path(get_subagent_projects_root())
    root = root.expanduser().resolve()
    sw._assert_root_isolated(root, repo_dir, sw._data_dir(data_dir))
    safe_task = project_folder_basename(dir_name) or sw._safe_name(task_id)
    # Genesis projects are durable: never clobber an existing entry. The candidate is
    # created EXCLUSIVELY before anything resolves it, so an existing directory, a
    # case/normalization alias, or a (dangling) symlink all count as taken -> "_N".
    root.mkdir(parents=True, exist_ok=True)
    _suffix = 0
    while True:
        proj = root / (f"{safe_task}_{_suffix}" if _suffix else safe_task)
        try:
            os.mkdir(proj)
            break
        except FileExistsError:
            _suffix += 1
    if proj.resolve().parent != root:
        raise RuntimeError(f"genesis project escaped its root: {proj}")
    proj = proj.resolve()
    try:
        sw._git(proj, "init")
        # A fresh repo may have no commit identity; set a local one for the seed
        # commit only (does not touch the user's global git config).
        sw._git(
            proj,
            "-c", "user.email=ouroboros@localhost",
            "-c", "user.name=Ouroboros",
            "commit", "--allow-empty", "-m", "genesis: empty project",
        )
        base_sha = sw._git(proj, "rev-parse", "HEAD").stdout.strip()
    except Exception:
        # Do not leak a partial/uninitialized project dir on git failure.
        sw._force_rmtree(proj)
        raise
    return sw.WorktreeHandle(
        task_id=str(task_id),
        path=str(proj),
        branch="",
        base_sha=base_sha,
        repo_dir=str(proj),
        created_at=time.time(),
        parent_task_id=str(parent_task_id or ""),
    )


def remove_genesis_project(path: str, *, projects_root: Optional[Any] = None, task_id: str = "",
                           data_dir: Optional[Any] = None) -> bool:
    """Best-effort removal of a provisioned-but-unused genesis project.

    Only removes a path strictly INSIDE the configured projects root (never an
    arbitrary caller path). Used to clean up a genesis project whose schedule was
    rejected before the child ran; genesis projects are otherwise durable. With
    ``task_id``, the owner's keep of that task (``owner_keep``) keeps it.
    """
    from ouroboros.config import get_subagent_projects_root
    from ouroboros import owner_keep

    if task_id:
        with owner_keep.gate(sw._data_dir(data_dir), task_id, timeout_sec=0) as why:
            return False if why else remove_genesis_project(path, projects_root=projects_root)

    if not str(path or "").strip():
        return False
    root = Path(projects_root) if projects_root else Path(get_subagent_projects_root())
    root = root.expanduser().resolve()
    target = Path(path).resolve()
    if target == root or not sw._is_within(target, root):
        return False
    if target.exists():
        sw._force_rmtree(target)
    return True
