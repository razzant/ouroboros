"""Workspace-task admission SSOT (v6.58.0, slice 1).

ONE validator + room-workspace resolver shared by the two surfaces that turn a
folder into a task's active workspace:

- ``gateway/tasks.py::api_tasks_create`` (the `/api/tasks` HTTP path), and
- ``supervisor/workers.py::promote_chat_to_task`` (the in-agent promote/route
  path — previously a DEGRADED twin that set ``workspace_root`` as a raw string
  with no validation).

Two invariants this module enforces (BIBLE P3/P5):

1. **One admission path.** Both surfaces call ``validate_workspace_root`` — the
   SAME git-worktree-root + repo/data-overlap check — so they cannot drift.
2. **Loud fail over silent self_modification.** A task born in a project ROOM
   whose ``working_dir`` is SET-but-unusable (deleted/moved/not a git worktree)
   must fail LOUDLY at admission, never silently run workspace-less — a
   workspace-less task resolves to the ``self_modification`` tool profile over the
   system repo (``tool_access.active_tool_profile``), which is exactly the danger
   the projects feature exists to steer work AWAY from.

The heavy per-task preflight (git snapshot + toolchain probes) stays on the
creation surface that can afford it: the async gateway handler runs it inline;
the promote path runs it under a hard time cap (``resolve_room_workspace`` does
only the cheap registry read + git-root validation, keeping the supervisor
event-drain thread responsive).
"""
from __future__ import annotations

import pathlib
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path


class WorkspaceRootError(ValueError):
    """A workspace_root that is missing, overlapping, or not a git worktree root."""


def validate_workspace_root(
    value: Any,
    *,
    system_repo_dir: Any,
    drive_root: Any,
) -> Optional[pathlib.Path]:
    """SSOT workspace-root validator (moved verbatim from gateway/tasks.py so both
    admission surfaces share it). Returns the resolved root, ``None`` for empty
    input, or raises ``WorkspaceRootError``: the path must exist, be a directory,
    NOT overlap the Ouroboros system repo or data drive, and BE the git worktree
    root (not a subdir of one)."""
    from ouroboros.tool_access import paths_overlap_casefold

    text = str(value or "").strip()
    if not text:
        return None
    root = pathlib.Path(text).expanduser().resolve(strict=False)
    system_repo = pathlib.Path(system_repo_dir).resolve(strict=False)
    drive = pathlib.Path(drive_root).resolve(strict=False)
    for protected_root, label in ((system_repo, "Ouroboros system repo"), (drive, "Ouroboros data drive")):
        overlaps = False
        try:
            root.relative_to(protected_root)
            overlaps = True
        except ValueError:
            try:
                protected_root.relative_to(root)
                overlaps = True
            except ValueError:
                pass
        if not overlaps and paths_overlap_casefold(root, protected_root):
            overlaps = True
        if overlaps:
            raise WorkspaceRootError(f"workspace_root must not overlap the {label}")
    if not root.exists() or not root.is_dir():
        raise WorkspaceRootError(f"workspace_root is not a directory: {text}")
    bootstrap_process_path()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        res = None
    git_root_text = (res.stdout or "").strip() if res is not None and res.returncode == 0 else ""
    git_root = pathlib.Path(git_root_text).resolve(strict=False) if git_root_text else None
    if git_root is None:
        raise WorkspaceRootError("workspace_root must be a git worktree root")
    if git_root != root:
        raise WorkspaceRootError(f"workspace_root must be the git worktree root: {git_root}")
    return root


# Sentinel: an explicit "no workspace" for a task born in a project room — distinct
# from an unset/empty value (which means "use the room's working_dir by default").
WORKSPACE_NONE = "none"


def resolve_room_workspace(
    *,
    drive_root: Any,
    system_repo_dir: Any,
    project_id: str,
    explicit_workspace: str = "",
    workspace_sentinel: str = "",
) -> tuple[str, str]:
    """Resolve the workspace_root for a task born in a project room (promote/route).

    Precedence (P5 — the semantic "this work belongs to project X" is already the
    LLM's/owner's decision; this only supplies the room's folder as transport):

    - ``workspace_sentinel == "none"`` → no workspace (explicit opt-out), returns ("","").
    - an ``explicit_workspace`` the caller passed → validated and used as-is.
    - else the project's registered ``working_dir`` (if any) → validated and used.
    - else no workspace (a file-less project), returns ("","").

    Returns ``(workspace_root, error)``. ``error`` is non-empty ONLY when a workspace
    was REQUESTED (explicit path or a set project working_dir) but is unusable — the
    caller MUST fail the task loudly rather than fall back to a workspace-less
    self_modification profile (the loud-fail invariant)."""
    if str(workspace_sentinel or "").strip().lower() == WORKSPACE_NONE:
        return "", ""

    requested = str(explicit_workspace or "").strip()
    source = "explicit workspace_root"
    if not requested and str(project_id or "").strip():
        try:
            from ouroboros.projects_registry import get_project

            project = get_project(drive_root, project_id) or {}
            requested = str(project.get("working_dir") or "").strip()
            source = f"project {project_id!r} working_dir"
        except Exception:
            requested = ""
    if not requested:
        return "", ""  # file-less project (or no working_dir): a non-workspace task

    try:
        resolved = validate_workspace_root(
            requested, system_repo_dir=system_repo_dir, drive_root=drive_root
        )
    except WorkspaceRootError as exc:
        # LOUD FAIL: a set-but-broken working_dir must never silently degrade to a
        # workspace-less (self_modification-profile) task over the system repo.
        return "", f"{source} is unusable: {exc}"
    return (str(resolved) if resolved else ""), ""


def room_chat_lens_dir(drive_root: Any, project_id: str) -> tuple[str, str]:
    """The project-room folder for the DIRECT-CHAT lens (v6.61.3), or ("", note).

    Chat-lane sibling of ``resolve_room_workspace``: the conversation lane of a
    folder-room re-points its reads/default-shell-cwd at the room folder so the
    tool affordance matches the room fact (the robot-room incident: ``.`` resolved
    to the system repo and the agent narrated the wrong tree). Requirements are
    LIGHTER than task admission — no git requirement (reading a plain folder in
    chat is fine); mutations still go through promoted tasks, which keep the full
    ``validate_workspace_root`` gate. Returns ``(dir, note)``: a set-but-unusable
    working_dir yields ("", loud note) so the chat context can disclose the
    breakage instead of silently falling back to the system repo."""
    pid = str(project_id or "").strip()
    if not pid:
        return "", ""
    try:
        from ouroboros.projects_registry import get_project

        project = get_project(drive_root, pid) or {}
        raw = str(project.get("working_dir") or "").strip()
    except Exception:
        return "", ""
    if not raw:
        return "", ""
    try:
        resolved = pathlib.Path(raw).expanduser().resolve(strict=False)
    except OSError as exc:
        return "", f"project {pid!r} working_dir is unusable: {type(exc).__name__}: {exc}"
    if not resolved.is_dir():
        return "", (
            f"project {pid!r} working_dir {raw} is unusable (missing or not a directory) — "
            "room reads/shell fall back to the system repo; fix or re-attach the folder"
        )
    return str(resolved), ""


def compose_workspace_block(
    *,
    workspace_root: Any,
    workspace_mode: str,
    memory_mode: str,
    workspace_preflight: dict,
) -> str:
    """The ``[HEADLESS_WORKSPACE]`` guidance block both admission surfaces embed in the
    task text (SSOT — previously gateway-only, so promoted room tasks ran with no
    workspace context at all). Returns the inner lines WITHOUT the wrapper markers."""
    from ouroboros.workspace_preflight import render_workspace_preflight_summary

    return (
        f"workspace_root: {workspace_root}\n"
        f"workspace_mode: {workspace_mode or 'external'}\n"
        f"memory_mode: {memory_mode}\n"
        "Use read_file, write_file, list_files, search_code, vcs_status, vcs_diff, and run_command against this target workspace, not the Ouroboros system repo.\n"
        f"{render_workspace_preflight_summary(workspace_preflight)}\n"
        "Before editing, account for target-repo docs or root-level instructions if present.\n"
        "Project-local dependency installs are allowed in external workspace tasks; system/global installs are for runtime_mode=pro only and must be noninteractive.\n"
        "When work naturally splits into independent branches, or while a long build/download/test is running, use schedule_subagent for a focused parallel handoff instead of serializing every branch yourself.\n"
        "Before finalizing, re-read the original task and verify each explicit requirement through the interface/path/format/service the task names; do not treat a weaker surrogate self-test as completion.\n"
        "Final summaries belong in the final answer, not new repo markdown files unless requested.\n"
        "Task-local git is allowed when the task requires it (clone, branch, commit, push to task-local remotes); "
        "Ouroboros still protects its own repo/data paths. Workspace artifacts are captured against the preflight git base.\n"
    )


def bounded_workspace_preflight(workspace_root: Any, *, timeout_sec: float = 8.0) -> dict:
    """Collect + summarize the workspace preflight under a HARD wall-clock cap.

    The promote path runs on the supervisor event-drain thread, which must stay
    responsive (the gateway path is an async handler and can afford the full run).
    The collection runs in a daemon thread; on timeout a DISCLOSED degraded summary
    is returned instead of blocking event delivery (P1 — the cut is visible, the
    task still admits). Never raises."""
    import threading

    result: dict = {}

    def _run() -> None:
        try:
            from ouroboros.workspace_preflight import (
                collect_workspace_preflight,
                summarize_workspace_preflight,
            )

            preflight = collect_workspace_preflight(pathlib.Path(str(workspace_root)))
            result["summary"] = summarize_workspace_preflight(preflight)
            result["preflight"] = preflight
        except Exception as exc:  # noqa: BLE001 — preflight is advisory context
            result["summary"] = {
                "schema_version": 1,
                "workspace_root": str(workspace_root),
                "error": f"{type(exc).__name__}: {exc}",
            }

    worker = threading.Thread(target=_run, name="room-workspace-preflight", daemon=True)
    worker.start()
    worker.join(timeout=max(1.0, float(timeout_sec)))
    if worker.is_alive() or "summary" not in result:
        return {
            "schema_version": 1,
            "workspace_root": str(workspace_root),
            "error": f"preflight exceeded {timeout_sec:.0f}s cap at admission; snapshot skipped (disclosed)",
        }
    return dict(result["summary"])
