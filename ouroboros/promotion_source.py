"""Supervisor-side source admission for promoted conversation work."""

from __future__ import annotations

import pathlib
from typing import Any, Tuple


def _source_project_id(source: str, is_git: bool) -> str:
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.project_sources import derive_repo_dir_name

    base = derive_repo_dir_name(source) if is_git else pathlib.Path(source.rstrip("/")).name
    return project_id_from_display_name(base or "project")


def resolve_promote_source(
    ctx: Any, source: str, project_id: str,
) -> Tuple[str, str, str, str]:
    """Attach/clone only after the supervisor has admitted an executor.

    Returns ``(workspace_root, note, error, effective_project_id)``.  Keeping
    this side effect on the authoritative handler side prevents a stale tool
    snapshot from cloning/registering a project after the worker pool was
    disabled but before the queued event was rejected.
    """
    from ouroboros.config import DATA_DIR
    from ouroboros.project_sources import (
        clone_project_repo,
        ephemeral_checkout_reason,
        valid_git_url,
        validate_attach_path,
    )

    src = str(source or "").strip()
    pid = str(project_id or "").strip()
    drive_root = pathlib.Path(getattr(ctx, "DRIVE_ROOT", DATA_DIR))
    if not src:
        return "", "", "", pid
    is_git = valid_git_url(src)
    pid = pid or _source_project_id(src, is_git)
    try:
        from ouroboros.projects_registry import get_reserved_project

        existing = get_reserved_project(drive_root, pid)
    except Exception as exc:
        return "", "", f"project_lookup_failed: {type(exc).__name__}: {exc}", pid
    lifecycle = str((existing or {}).get("lifecycle") or "active")
    if existing is not None and lifecycle != "active":
        return "", "", f"project_routing_fence: {pid!r} is {lifecycle}", pid
    if is_git:
        if str((existing or {}).get("working_dir") or "").strip():
            return "", "", (
                f"conflict: project {pid!r} already has folder {existing.get('working_dir')}; "
                "use another project id or omit source"
            ), pid
        cloned, code, detail = clone_project_repo(src, pid)
        if code:
            return "", "", f"{code}: {detail}", pid
        folder, provenance, clone_url = cloned, "cloned", src
        note = f"cloned {src} -> {cloned}"
    else:
        # A11/A12: the folder becomes the project's PLACE on the strength of the
        # safety guards alone (exists, real dir, not the home root, disjoint from
        # the Ouroboros repo/data roots). Being a git worktree is NOT one of them
        # any more — an untracked folder is admitted and STAYS untracked until the
        # owner answers the typed `git_init_required` offer that task admission
        # raises before any file work is queued (workspace_admission).
        resolved, err = validate_attach_path(
            src,
            system_repo_dir=getattr(ctx, "REPO_DIR", getattr(ctx, "repo_dir", "")),
            drive_root=drive_root,
        )
        if err:
            return "", "", f"attach: {err}", pid
        # The DURABLE-place rule, not just the attach guards. `source` here is typed
        # by an AGENT, not by the owner clicking a folder, and the paths an agent has
        # in hand are exactly the checkouts Ouroboros makes for itself: a linked
        # worktree, a subagent `self_worktree`, a thread's branch-off. Any of those
        # becomes a project's permanent home that a `git worktree remove` or the
        # orphan sweep can delete underneath it. `adopt_task_workspace` applies this
        # rule for the same reason; this surface needs it MORE, because no owner ever
        # looked at the path.
        ephemeral = ephemeral_checkout_reason(resolved)
        if ephemeral:
            return "", "", f"attach: {ephemeral}", pid
        folder, provenance, clone_url = str(resolved), "attached", ""
        note = f"attached {resolved}"
    prior_wd = str((existing or {}).get("working_dir") or "").strip()
    if prior_wd and prior_wd != folder:
        return "", "", (
            f"conflict: project {pid!r} already has folder {prior_wd}; use another project id "
            "or omit source"
        ), pid
    if prior_wd == folder and str((existing or {}).get("provenance") or "").strip() not in ("", "none"):
        return folder, note, "", pid
    try:
        from ouroboros.projects_registry import create_project, update_project
        from ouroboros.utils import utc_now_iso

        create_project(drive_root, pid, origin="promote_chat_to_task")
        update_project(
            drive_root,
            pid,
            working_dir=folder,
            provenance=provenance,
            clone_url=clone_url,
            trusted_at=utc_now_iso(),
        )
    except Exception as exc:
        return "", "", f"register: {type(exc).__name__}: {exc}", pid
    return folder, note, "", pid


__all__ = ["resolve_promote_source"]
