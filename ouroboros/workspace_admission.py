"""Workspace-task admission SSOT (v6.58.0, slice 1).

ONE validator + room-workspace resolver shared by the two surfaces that turn a
folder into a task's active workspace:

- ``gateway/tasks.py::api_tasks_create`` (the `/api/tasks` HTTP path), and
- ``supervisor/workers.py::promote_chat_to_task`` (the in-agent promote/route
  path — previously a DEGRADED twin that set ``workspace_root`` as a raw string
  with no validation).

Three invariants this module enforces (BIBLE P3/P5):

1. **One admission path.** Both surfaces call ``validate_workspace_root`` — the
   SAME git-worktree-root + repo/data-overlap check — so they cannot drift.
2. **Loud fail over silent self_modification.** A task born in a project ROOM
   whose ``working_dir`` is SET-but-unusable (deleted/moved/not a git worktree)
   must fail LOUDLY at admission, never silently run workspace-less — a
   workspace-less task resolves to the ``self_modification`` tool profile over the
   system repo (``tool_access.active_tool_profile``), which is exactly the danger
   the projects feature exists to steer work AWAY from.
3. **Git is OFFERED, never forced (A12).** A plain, safe folder that is simply not
   tracked by git is no longer one more unusable path: admission still stops before
   the task is queued — auto-``git init`` in someone else's folder stays forbidden —
   but it stops with the TYPED ``git_init_required`` decision
   (``GitInitRequiredError.decision``) that the owner answers, instead of an error
   they have to decode. The project itself keeps the folder either way; the folder
   is the project's PLACE (A11), and only FILE WORK waits on the answer.

The heavy per-task preflight (git snapshot + toolchain probes) stays on the
creation surface that can afford it: the async gateway handler runs it inline;
the promote path runs it under a hard time cap (``resolve_room_workspace`` does
only the cheap registry read + git-root validation, keeping the supervisor
event-drain thread responsive).
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path

log = logging.getLogger(__name__)


class WorkspaceRootError(ValueError):
    """A workspace_root that is missing, overlapping, or not a git worktree root."""


# The one spelling of the decision, shared by the exception, the gateway error
# envelope and the promote outcome so the three cannot drift apart.
GIT_INIT_REQUIRED = "git_init_required"


def git_init_decision(workspace_root: Any, *, project_id: str = "") -> dict:
    """The typed ``git_init_required`` decision (A12) — an OFFER, not a refusal.

    Returned INSTEAD of queueing a file task in a folder that is safe and valid but
    not tracked by git. It carries its own plain-language reason so every surface
    (gateway 400 body, promote outcome, chat message) says the same honest thing
    about what git buys, and no surface has to invent copy of its own.

    The message names WHO runs the offer, because the agent reads it too. Shell
    policy permits ``git init`` in an attached project folder (it protects the
    Ouroboros runtime, not the owner's tree), so a halt message that only said
    "Ouroboros can start tracking it" invited the agent to execute the owner's yes
    on its behalf — precisely the auto-init A12 forbids.
    """
    return {
        "decision": GIT_INIT_REQUIRED,
        "workspace_root": str(workspace_root),
        "project_id": str(project_id or ""),
        "offer": "init_git",
        "enables": ["diff", "rollback", "branching"],
        "message": (
            f"{workspace_root} is not tracked by git, so file work there cannot be "
            "diffed, rolled back, or branched. Saying yes runs one snapshot commit of "
            "what is already in the folder, with credential-shaped files deliberately "
            "left untracked. Only you can do this — Ouroboros will not run `git init` "
            "here. Nothing is initialised until you say yes: the folder is yours."
        ),
    }


class GitInitRequiredError(WorkspaceRootError):
    """The folder is a SAFE, valid workspace that is simply not git-backed (A12).

    Kept a subclass of ``WorkspaceRootError`` on purpose: a caller that knows
    nothing about the offer still refuses admission exactly as it did before, so
    removing the git requirement can never quietly admit a file task. A caller that
    CAN ask the owner catches this first and renders ``decision``. The validator
    knows only the path, so ``decision`` carries no ``project_id``; the room-level
    caller re-stamps it with ``git_init_decision`` once it knows whose folder it is.
    """

    def __init__(self, root: Any) -> None:
        self.decision = git_init_decision(root)
        super().__init__(str(self.decision["message"]))


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
    root (not a subdir of one).

    A folder that passes every safety guard and is merely UNTRACKED raises the
    ``GitInitRequiredError`` subclass instead, carrying the owner's offer (A12).
    A folder INSIDE someone else's worktree stays a plain refusal — initialising
    git there would nest a second repository, which is not an offer worth making."""
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
        raise GitInitRequiredError(root)
    if git_root != root:
        raise WorkspaceRootError(f"workspace_root must be the git worktree root: {git_root}")
    return root


# Sentinel: an explicit "no workspace" for a task born in a project room — distinct
# from an unset/empty value (which means "use the room's working_dir by default").
WORKSPACE_NONE = "none"


def thread_checkout_for_room(drive_root: Any, project_id: str, room_chat_id: Any) -> tuple[str, str]:
    """The registered CHECKOUT of the thread this room is, or ``("", "")``.

    Returns ``(path, label)``; ``label`` names the source for a loud failure.

    This is the binding that makes branching off mean anything (A7/A14/R1). The
    writer lane is keyed on the FOLDER, so two tasks run at once exactly when they
    name two folders — but nothing connected a THREAD's checkout to the workspace
    its tasks are admitted into: ``resolve_room_workspace`` read the project's
    ``working_dir`` and ``get_thread_worktree`` had no consumer in any admission
    path at all. So a branched thread's task took the project-folder lane and
    QUEUED behind the main thread's task, while ``queue_notice`` — which keys its
    candidate on ``thread_location(...)["path"]`` — answered "this will not wait"
    and the branch-off copy promised "both can run at the same time". The two
    surfaces disagreed, and the one the owner was reading was the wrong one.

    Deliberately narrow: the room must resolve to a thread of THIS project, or the
    answer is nothing. A chat id that belongs somewhere else is not this task's
    workspace, and a mismatch here would move a writer into another project's
    checkout rather than merely failing to help.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return "", ""
    try:
        chat_id = int(room_chat_id or 0)
    except (TypeError, ValueError):
        return "", ""
    if not chat_id:
        return "", ""
    try:
        from ouroboros.projects_registry import resolve_chat_binding
        from ouroboros.thread_worktrees import get_thread_worktree

        binding = resolve_chat_binding(drive_root, chat_id)
        if str(binding.get("project_id") or "") != pid:
            return "", ""
        thread_id = int(binding.get("thread_id") or 0)
        row = get_thread_worktree(drive_root, pid, thread_id) or {}
    except Exception:
        # A registry read that fails is NOT "this thread works in the project
        # folder": answering that would silently put the writer back in the folder
        # branching off exists to keep it out of. The caller loud-fails instead.
        log.warning(
            "thread_checkout_for_room: registry read failed for %r chat %r",
            pid, room_chat_id, exc_info=True,
        )
        return "", "unreadable"
    path = str(row.get("path") or "").strip()
    if not path:
        return "", ""
    return path, f"project {pid!r} thread {thread_id} checkout"


def resolve_room_workspace(
    *,
    drive_root: Any,
    system_repo_dir: Any,
    project_id: str,
    explicit_workspace: str = "",
    workspace_sentinel: str = "",
    room_chat_id: Any = 0,
) -> tuple[str, str, dict]:
    """Resolve the workspace_root for a task born in a project room (promote/route).

    Precedence (P5 — the semantic "this work belongs to project X" is already the
    LLM's/owner's decision; this only supplies the room's folder as transport):

    - ``workspace_sentinel == "none"`` → no workspace (explicit opt-out), returns ("","").
    - an ``explicit_workspace`` the caller passed → validated and used as-is.
    - else the CHECKOUT registered for the room's own thread, if it has one (A7:
      that is what branching off did, and it is the only thing that makes the
      thread a second writer lane instead of a second queue entry).
    - else the project's registered ``working_dir`` (if any) → validated and used.
    - else no workspace (a file-less project), returns ("","").

    ``room_chat_id`` is the room the task was born in — a thread's chat id. It is
    how this function knows WHICH thread is asking; without it the answer is the
    project's folder for every thread of the project, branched or not.

    Returns ``(workspace_root, error, decision)``. ``error`` is non-empty when a
    workspace was REQUESTED (explicit path or a set project working_dir) but is
    unusable, AND when the project's registry entry cannot be READ at all — the
    caller MUST fail the task loudly rather than fall back to a workspace-less
    self_modification profile (the loud-fail invariant). ``decision`` is the typed
    ``git_init_required`` OFFER (A12) for the one case that is not a breakage: a
    perfectly good folder nobody has put under git yet. It is a THIRD outcome on
    purpose — collapsing it into ``error`` would present the owner's open choice as
    someone's mistake, and collapsing it into a resolved root would queue file work
    with no diff, no rollback and no way back."""
    if str(workspace_sentinel or "").strip().lower() == WORKSPACE_NONE:
        return "", "", {}

    requested = str(explicit_workspace or "").strip()
    source = "explicit workspace_root"
    if not requested and str(project_id or "").strip():
        # A7's whole payoff: a thread that BRANCHED OFF works in its own checkout,
        # so its tasks must be admitted into that folder — otherwise they take the
        # project folder's writer lane and queue behind it, and branching bought
        # the owner nothing but a second copy of their files.
        checkout, checkout_label = thread_checkout_for_room(drive_root, project_id, room_chat_id)
        if checkout_label == "unreadable":
            return "", (
                f"the thread-worktree registry for project {project_id!r} is unreadable "
                "— cannot tell whether this thread works in its own checkout"
            ), {}
        if checkout:
            requested, source = checkout, checkout_label
    if not requested and str(project_id or "").strip():
        try:
            from ouroboros.projects_registry import get_project

            project = get_project(drive_root, project_id) or {}
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
            ), {}
        requested = str(project.get("working_dir") or "").strip()
        source = f"project {project_id!r} working_dir"
    if not requested:
        return "", "", {}  # file-less project (or no working_dir): a non-workspace task

    try:
        resolved = validate_workspace_root(
            requested, system_repo_dir=system_repo_dir, drive_root=drive_root
        )
    except GitInitRequiredError as exc:
        # NOT a failure: the folder is fine, it is simply untracked. Hand the owner
        # the offer and let admission stop here (A12 — never auto-init).
        return "", "", git_init_decision(exc.decision["workspace_root"], project_id=str(project_id or ""))
    except WorkspaceRootError as exc:
        # LOUD FAIL: a set-but-broken working_dir must never silently degrade to a
        # workspace-less (self_modification-profile) task over the system repo.
        return "", f"{source} is unusable: {exc}", {}
    return (str(resolved) if resolved else ""), "", {}


def room_chat_lens_dir(
    drive_root: Any, project_id: str, room_chat_id: Any = 0,
) -> tuple[str, str]:
    """The project-room folder for the DIRECT-CHAT lens (v6.61.3), or ("", note).

    Chat-lane sibling of ``resolve_room_workspace``: the conversation lane of a
    folder-room re-points its reads/default-shell-cwd at the room folder so the
    tool affordance matches the room fact (the robot-room incident: ``.`` resolved
    to the system repo and the agent narrated the wrong tree). Requirements are
    LIGHTER than task admission — no git requirement (reading a plain folder in
    chat is fine); mutations still go through promoted tasks, which keep the full
    ``validate_workspace_root`` gate. Returns ``(dir, note)``: a set-but-unusable
    working_dir yields ("", loud note) so the chat context can disclose the
    breakage instead of silently falling back to the system repo.

    ``room_chat_id`` is the room asking, and it applies the SAME precedence
    ``resolve_room_workspace`` does: the checkout registered for THAT thread
    first, the project's ``working_dir`` otherwise. Without it this answered the
    project folder unconditionally, so a branched thread's TASKS wrote its
    checkout while its CHAT read the project folder and the model was told the
    promoted task would inherit the folder it was looking at — the same
    fact/affordance split the robot-room incident was (I7). It is a keyword with
    a default because both call sites hold the id and a project-wide caller may
    not; a caller that omits it gets the pre-thread answer, which is correct only
    for an unbranched thread."""
    pid = str(project_id or "").strip()
    if not pid:
        return "", ""
    checkout, label = thread_checkout_for_room(drive_root, pid, room_chat_id)
    if label == "unreadable":
        return "", (
            f"the thread-worktree registry for project {pid!r} is unreadable — cannot "
            "tell whether this room works in its own checkout; room reads/shell fall "
            "back to the system repo"
        )
    raw = checkout
    what = f"thread checkout {checkout}" if checkout else ""
    if not raw:
        try:
            from ouroboros.projects_registry import get_project

            project = get_project(drive_root, pid) or {}
            raw = str(project.get("working_dir") or "").strip()
        except Exception:
            return "", ""
        what = f"working_dir {raw}"
    if not raw:
        return "", ""
    try:
        resolved = pathlib.Path(raw).expanduser().resolve(strict=False)
    except OSError as exc:
        return "", f"project {pid!r} {what} is unusable: {type(exc).__name__}: {exc}"
    if not resolved.is_dir():
        return "", (
            f"project {pid!r} {what} is unusable (missing or not a directory) — "
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
        "Ouroboros still protects its own repo/data paths. One exception: never run `git init` in the owner's "
        "project folder — only the owner can put their folder under git, through the git_init_required offer. "
        "Workspace artifacts are captured against the preflight git base.\n"
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
