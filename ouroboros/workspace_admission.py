"""Workspace-task admission SSOT (v6.58.0, slice 1).

ONE validator + room-workspace resolver shared by the surfaces that turn a
folder into a task's active workspace:

- ``gateway/tasks.py::api_tasks_create`` (the `/api/tasks` HTTP path),
- ``supervisor/workers.py::promote_chat_to_task`` (the in-agent promote/route
  path — previously a DEGRADED twin that set ``workspace_root`` as a raw string
  with no validation), and
- owner-selected Presence folders (local configuration and turn admission).

Two invariants this module enforces (BIBLE P3/P5):

1. **One admission path.** These surfaces call ``validate_workspace_root`` — the
   SAME folder + Git geometry + repo/data-overlap check — so they cannot drift.
2. **Loud fail over silent self_modification.** A task born in a project ROOM
   whose ``working_dir`` is SET-but-unusable (deleted/moved/invalid Git root)
   must fail LOUDLY at admission, never silently run workspace-less — a
   workspace-less task resolves to the ``self_modification`` tool profile over the
   system repo (``tool_access.active_tool_profile``), which is exactly the danger
   the projects feature exists to steer work AWAY from.

The heavy per-task preflight (git snapshot + toolchain probes) stays on the
creation surface that can afford it: the async gateway handler runs it inline;
the promote path runs it under a hard time cap (``resolve_room_workspace`` does
only the cheap registry read + folder validation, keeping the supervisor
event-drain thread responsive).
"""
from __future__ import annotations

import logging
import os
import pathlib
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path

log = logging.getLogger(__name__)


class WorkspaceRootError(ValueError):
    """A workspace_root that is missing, overlapping, or has invalid Git geometry."""


def has_git_metadata(root: pathlib.Path) -> bool:
    """Match Git's ancestor search, including an explicitly selected ceiling."""
    ceilings = {pathlib.Path(p).resolve() for p in
                os.environ.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep) if p}
    for parent in (root, *root.parents):
        if parent != root and parent in ceilings:
            break
        if (parent / ".git").exists() or (parent / ".git").is_symlink():
            return True
    return False


def validate_workspace_root(
    value: Any,
    *,
    system_repo_dir: Any,
    drive_root: Any,
) -> Optional[pathlib.Path]:
    """Return an existing ordinary folder or Git worktree root, ``None`` for
    empty input, or raise ``WorkspaceRootError``. Git is optional, but an
    existing repository must resolve to its worktree root, not a subdirectory.
    Repo/data overlap and broken addresses never become workspace-less tasks.
    """
    text = str(value or "").strip()
    if not text:
        return None
    root = pathlib.Path(text).expanduser().resolve(strict=False)
    _refuse_protected_overlap(root, system_repo_dir=system_repo_dir, drive_root=drive_root)
    if not root.exists() or not root.is_dir():
        raise WorkspaceRootError(f"workspace_root is not a directory: {text}")
    bootstrap_process_path()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-bare-repository", "--show-toplevel"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        res = None
    output = (res.stdout or "").strip() if res is not None else ""
    if output.splitlines()[:1] == ["true"]:
        raise WorkspaceRootError("workspace_root must be a Git worktree, not a bare repository")
    git_root_text = output.partition("\n")[2] if res is not None and res.returncode == 0 else ""
    git_root = pathlib.Path(git_root_text).resolve(strict=False) if git_root_text else None
    if git_root is None:
        # A failed probe must not reclassify an existing/broken Git worktree
        # as an ordinary folder (including linked worktrees with a .git file).
        if has_git_metadata(root):
            raise WorkspaceRootError("workspace_root Git worktree could not be resolved")
        return root
    if git_root != root:
        raise WorkspaceRootError(f"workspace_root must be the git worktree root: {git_root}")
    return root


def _refuse_protected_overlap(root: pathlib.Path, *, system_repo_dir: Any, drive_root: Any) -> None:
    from ouroboros.tool_access import paths_overlap_casefold

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


def unavailable_workspace_address(value: Any, *, system_repo_dir: Any, drive_root: Any) -> str:
    """The resolved address of a selected folder that is merely absent now, else ``""``.

    Work that keeps a known folder (#1799, owner choice A) starts with that address
    instead of waiting or falling back, as a direct room keeps its missing folder
    (``room_chat_lens_dir``). Only absence qualifies: the repo/data overlap refusal of
    ``validate_workspace_root`` still applies, and anything at the path (a file, a bare
    or non-root Git tree) keeps its own refusal.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    root = pathlib.Path(text).expanduser().resolve(strict=False)
    try:
        _refuse_protected_overlap(root, system_repo_dir=system_repo_dir, drive_root=drive_root)
    except WorkspaceRootError:
        return ""
    return "" if root.exists() or root.is_symlink() else str(root)


def admitted_absent_workspace(task: Any, *, system_repo_dir: Any, drive_root: Any) -> bool:
    """Whether ``task`` was admitted with its kept folder absent (``workspace_availability``)
    and that exact address is still merely absent, so restore revalidation keeps the original
    admission instead of holding the row for that folder. Anything now at the path, another
    resolution or a protected overlap answers ``False`` and keeps the prepared-folder check."""
    row = task if isinstance(task, dict) else {}
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    availability = meta.get("workspace_availability")
    workspace = str(row.get("workspace_root") or "")
    if not workspace or not isinstance(availability, dict) or availability.get("state") != "missing":
        return False
    current = unavailable_workspace_address(workspace, system_repo_dir=system_repo_dir, drive_root=drive_root)
    return bool(current) and os.path.normcase(current) == os.path.normcase(workspace)


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
    project_admission: Optional[dict] = None,
) -> tuple[str, str]:
    """Resolve the workspace_root for a task born in a project room (promote/route).

    Precedence (P5 — the semantic "this work belongs to project X" is already the
    LLM's/owner's decision; this only supplies the room's folder as transport):

    - ``workspace_sentinel == "none"`` → no workspace (explicit opt-out), returns ("","").
    - an ``explicit_workspace`` the caller passed → validated and used as-is.
    - else the project's registered ``working_dir`` (if any) → validated and used.
    - else no workspace (a file-less project), returns ("","").

    Returns ``(workspace_root, error)``. ``error`` is non-empty when a workspace was
    REQUESTED (explicit path or a set project working_dir) but is unusable, AND when
    the project's registry entry cannot be READ at all — the caller MUST fail the task
    loudly rather than fall back to a workspace-less self_modification profile (the
    loud-fail invariant)."""
    if str(workspace_sentinel or "").strip().lower() == WORKSPACE_NONE:
        return "", ""

    requested = str(explicit_workspace or "").strip()
    source = "explicit workspace_root"
    if not requested and str(project_id or "").strip():
        try:
            from ouroboros.projects_registry import project_admission_view

            view = project_admission if project_admission is not None else project_admission_view(drive_root, project_id)
            project = view["project"]
            if project is None:
                raise ValueError("Registered Project is missing")
        except Exception as exc:
            # BIND-OR-LOUD-FAIL. "The registry could not be read" is NOT the same
            # fact as "this project has no working_dir", and collapsing the two was a
            # silent re-entry of the very regression this SSOT exists to kill: the
            # swallowed exception returned ("", "") — indistinguishable from a
            # file-less project — so admission continued and the task ran
            # workspace-less on the self_modification profile over the SYSTEM repo.
            # An unreadable registry is an error; the caller fails the task loudly.
            log.warning(
                "resolve_room_workspace: project registry read failed for %r", project_id, exc_info=True
            )
            return "", (
                f"project {project_id!r} registry entry is unreadable "
                f"({type(exc).__name__}: {exc}) — cannot determine the task's workspace"
            )
        requested = str(project.get("working_dir") or "").strip()
        source = f"project {project_id!r} working_dir"
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


def workspace_repair_hint(
    *,
    ws_error: str,
    explicit_workspace: str = "",
    project_id: str = "",
    project_folder: str = "",
    presence: bool = False,
    retired_worktree: bool = False,
    drive_root: Any = None,
    system_repo_dir: Any = None,
) -> str:
    """The MODEL-facing repair for one refused workspace: the typed cause plus
    the one move that fixes it, following the SOURCE of the refused folder.

    ``resolve_room_workspace`` already types the source, so the repair follows
    it instead of sending the caller to a Projects setting the failure never
    read: a Presence profile's folder is fixed in the profile; a path the
    REQUEST named is re-promoted against the project's folder or with
    ``workspace='none'`` (a subfolder of the Ouroboros repository can never be a
    workspace, and a delegated-run worktree is gone once its run ends); a
    project ``working_dir`` — or a failed auto-provision — is fixed in Projects.
    The project folder is named only when the registry can be read; the
    worktree/repo facts are derived here unless the caller already knows them.
    Never raises; this text rides ``detail`` into the typed refusal.
    """
    cause = str(ws_error or "").strip().rstrip(".")
    if presence:
        return (
            f"The folder configured in the Presence profile is unusable: {cause}. "
            "Fix the Presence profile's workspace_root or clear it."
        )
    explicit = str(explicit_workspace or "").strip()
    if not explicit:
        return (
            f"{cause}. Fix the project's working folder (Projects → this project) "
            "or re-promote with workspace='none' for a folder-less task."
        )
    requested = pathlib.Path(explicit).expanduser()
    if system_repo_dir is not None:
        from ouroboros.tool_access import paths_overlap_casefold

        try:
            under_repo = paths_overlap_casefold(requested, pathlib.Path(system_repo_dir))
        except Exception:
            under_repo = False
            log.debug("workspace repair hint: repo-overlap check failed for %r", explicit, exc_info=True)
        if under_repo:
            return (
                f"{cause}. workspace_root must be a folder outside the Ouroboros repository, "
                "or empty (or workspace='none') to work in the repository itself."
            )
    if not retired_worktree:
        try:
            from ouroboros.config import get_subagent_worktree_root
            from ouroboros.tool_access_paths import path_is_relative_to

            retired_worktree = path_is_relative_to(requested, pathlib.Path(get_subagent_worktree_root()))
        except Exception:
            log.debug("workspace repair hint: worktree-root check failed for %r", explicit, exc_info=True)
    folder = str(project_folder or "").strip()
    if not folder and str(project_id or "").strip() and drive_root is not None:
        try:
            from ouroboros.projects_registry import get_project

            folder = str((get_project(drive_root, project_id) or {}).get("working_dir") or "").strip()
        except Exception:
            log.debug("workspace repair hint: project working_dir unreadable for %s", project_id, exc_info=True)
    named_folder = f" ({folder})" if folder else ""
    if retired_worktree:
        return (
            f"{cause}. That path is inside a delegated-run worktree, which is removed when "
            f"its run ends; re-promote with the project's folder{named_folder} or with workspace='none'."
        )
    return (
        f"{cause}. This task asked for {explicit} explicitly; re-promote it against the "
        f"project folder{named_folder} or with workspace='none' for a folder-less task."
    )


def room_chat_lens_dir(drive_root: Any, project_id: str) -> tuple[str, str]:
    """The selected folder for a direct conversation, with an availability note.

    Ordinary folders need no Git admission. A missing folder keeps its selected
    address so tools cannot fall back to the system repository; an unreadable
    registry instead returns a note without inventing an address.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return "", ""
    try:
        from ouroboros.projects_registry import get_reserved_project

        # Strict: an unreadable registry is a note, never a folderless room (#1315).
        project = get_reserved_project(drive_root, pid, strict=True) or {}
        if str(project.get("lifecycle") or "active") != "active":
            project = {}
        raw = str(project.get("working_dir") or "").strip()
    except Exception as exc:
        return "", (
            f"project {pid!r} registry entry is unreadable ({type(exc).__name__}: {exc}) — "
            "cannot determine the room's repository"
        )
    if not raw:
        return "", ""
    try:
        resolved = pathlib.Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as exc:
        return "", f"project {pid!r} working_dir is unusable: {type(exc).__name__}: {exc}"
    if not resolved.is_dir():
        return str(resolved), (
            f"project {pid!r} working_dir {raw} is unusable (missing or not a directory) — "
            "the selected folder remains the active target; fix or re-attach the folder"
        )
    return str(resolved), ""


def room_lens_applies(task: dict) -> bool:
    """Whether a room's folder (``room_chat_lens_dir``) is this direct turn's default.

    A room default never replaces a resource the work keeps: its own workspace, or a
    recorded ``system_repo``/``explicit_none`` intent. Ordinary direct turns record no
    intent; readmission records ``system_repo`` for placed body work (``project_placement``).
    """
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    intent = metadata.get("resource_intent") if isinstance(metadata.get("resource_intent"), dict) else {}
    return (bool(task.get("_is_direct_chat")) and not str(task.get("workspace_root") or "").strip()
            and intent.get("kind") not in {"system_repo", "explicit_none"})


def compose_workspace_block(
    *,
    workspace_root: Any,
    workspace_mode: str,
    memory_mode: str,
    workspace_preflight: dict,
    unavailable: str = "",
) -> str:
    """The ``[HEADLESS_WORKSPACE]`` guidance block both admission surfaces embed in the
    task text (SSOT — previously gateway-only, so promoted room tasks ran with no
    workspace context at all). Returns the inner lines WITHOUT the wrapper markers.
    ``unavailable`` names why a kept address is absent (``unavailable_workspace_address``)."""
    from ouroboros.workspace_preflight import render_workspace_preflight_summary

    return (
        f"workspace_root: {workspace_root}\n"
        + (f"workspace_availability: missing when this work was admitted ({unavailable}). It is still this "
           "work's selected folder and nothing else was substituted; commands that run inside it fail "
           "until it exists again.\n" if unavailable else "")
        + f"workspace_mode: {workspace_mode or 'external'}\n"
        f"memory_mode: {memory_mode}\n"
        "Use read_file, write_file, list_files, search_code, vcs_read, and run_command against this target workspace, not the Ouroboros system repo.\n"
        f"{render_workspace_preflight_summary(workspace_preflight)}\n"
        "Before editing, account for target-repo docs or root-level instructions if present.\n"
        "Project-local dependency installs are allowed in external workspace tasks; system/global installs are for runtime_mode=pro or cyber_pro and must be noninteractive.\n"
        "When work naturally splits into independent branches, or while a long build/download/test is running, use schedule_subagent for a focused parallel handoff instead of serializing every branch yourself.\n"
        "Before finalizing, re-read the original task and verify each explicit requirement through the interface/path/format/service the task names; do not treat a weaker surrogate self-test as completion.\n"
        "Final summaries belong in the final answer, not new repo markdown files unless requested.\n"
        "Task-local git is allowed when the task requires it (clone, branch, commit, push to task-local remotes); "
        "Ouroboros still protects its own repo/data paths. Git results use the preflight base; "
        "ordinary-folder work stays on site with captured output references, without a Git patch or full rollback promise.\n"
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
