"""Workspace-task admission SSOT (v6.58.0, slice 1; RWS v2 §3.3 discriminated refs).

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

**RWS v2 — discriminated admission, NO new state authority.** ``validate_workspace_root``
is polymorphic over the REQUESTED placement and returns a sealed
:data:`~ouroboros.workspace_ref.WorkspaceRef`, not a path:

* **local** — the pre-existing checks run BYTE-IDENTICALLY (overlap with the
  Ouroboros repo/data drive first, then existence/kind, then
  ``git rev-parse --show-toplevel`` equality), and the accepted path is sealed as
  a ``LocalWorkspaceRef``. The probe stays inline here (not behind the
  ``execution_facts`` door) precisely so this branch keeps its exact process/PATH
  behavior.
* **ssh** — the ref FORM is validated fully and locally (absolute normalized
  POSIX ``remote_root`` with no traversal, non-empty connection/workspace
  identity, no unknown fields — ``workspace_ref.normalize_workspace_ref``; plus
  ``connection_id`` must be known to the connection store), and the TARGET facts
  are read through the placement-neutral fact door. Until Lane 1's transport
  lands, that door refuses typed with :data:`REMOTE_TRANSPORT_UNAVAILABLE` — an
  ssh workspace is NEVER admitted on Home evidence, and never degrades to a Home
  path (the "no silent fallback" invariant).

Durable task transitions are NOT this module's business: they stay in
``supervisor/queue.py`` under ``_queue_lock`` (helpers in
``supervisor/task_lifecycle.py``). This module only decides WHAT the placement is
and seals it; the queue decides WHEN a task becomes PENDING and revalidates the
placement fence immediately before insertion.
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
from typing import Any, Optional

from ouroboros.platform_layer import bootstrap_process_path
from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS
from ouroboros.workspace_ref import (
    LocalWorkspaceRef,
    SshWorkspaceRef,
    WorkspaceRef,
    normalize_remote_root,
    normalize_workspace_ref,
)

# Typed refusal code for the ssh admission branch when the TARGET fact source is
# not reachable. Distinct from `execution_facts.SSH_FACTS_UNAVAILABLE` (prepare
# phase) and `workspace_executor.SSH_EXECUTOR_UNAVAILABLE` (execute phase) on
# purpose: a diagnosis must be able to name WHICH phase refused, and admission is
# the phase that decides whether a task exists at all.
REMOTE_TRANSPORT_UNAVAILABLE = "REMOTE_TRANSPORT_UNAVAILABLE"

log = logging.getLogger(__name__)


class WorkspaceRootError(ValueError):
    """A workspace_root that is missing, overlapping, or not a git worktree root."""


class RemoteWorkspaceUnavailableError(WorkspaceRootError):
    """The ssh ref is well-FORMED but its target facts cannot be reached yet.

    Subclasses ``WorkspaceRootError`` (hence ``ValueError``) so every existing
    ``except ValueError`` admission call site keeps refusing loudly; ``code``
    lets a caller distinguish "your request is malformed" from "this placement
    is not serviceable right now".

    It also carries an ``action``, because this class is where a derived one used
    to DIE. The broker's refusal arrives as a ``RemoteWorkspaceError`` that knows
    exactly what removes it (an outdated executor wants Bootstrap, a changed host
    identity wants Retrust, a stale bundle wants a rebuild); the wrapper flattened
    it into a message string, and both HTTP callers then answered a hardcoded
    ``action: "bootstrap_connection"`` — which is the right advice for one of those
    three and a dead end for the other two. ``cause`` is the wrapped refusal, so
    the action travels instead of being re-guessed at the boundary.
    """

    code = REMOTE_TRANSPORT_UNAVAILABLE

    def __init__(
        self,
        detail: str = "",
        *,
        action: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.action = str(
            action
            or getattr(cause, "action", "")
            or REFUSAL_ACTIONS.get(REMOTE_TRANSPORT_UNAVAILABLE.lower())
            or "retry"
        )
        super().__init__(f"{REMOTE_TRANSPORT_UNAVAILABLE}: {detail}" if detail else REMOTE_TRANSPORT_UNAVAILABLE)


def known_connections() -> Optional[dict[str, str]]:
    """``{connection_id: trust identity}`` from the owner-state store, or ``None``.

    ONE door onto the connection store, deliberately: membership (does this
    connection exist?) and trust identity (is it still the SAME trusted host?) are
    the two facts admission needs, and reading them through two doors is how they
    start disagreeing. ``None`` means "the remote surface is not installed in this
    build" — the ssh branch then refuses with :data:`REMOTE_TRANSPORT_UNAVAILABLE`
    rather than admitting an unverifiable connection identity.

    The trust identity is an OPAQUE string: admission compares it for equality and
    never interprets it, so the store owns what "the same trusted host" means.
    It takes no drive argument because D6 gives ``connection_store`` sole ownership
    of the store's path — a drive parameter here would be a second place that
    decides where owner state lives.
    """
    try:
        from ouroboros.connection_store import connection_trust_index
    except Exception:
        return None
    try:
        index = connection_trust_index() or {}
        return {str(key): str(value or "") for key, value in dict(index).items() if str(key or "").strip()}
    except Exception:
        return {}


def _admit_local_workspace_root(
    value: Any,
    *,
    system_repo_dir: Any,
    drive_root: Any,
) -> Optional[pathlib.Path]:
    """The LOCAL admission branch, unchanged since v6.58.0: the path must exist, be a
    directory, NOT overlap the Ouroboros system repo or data drive, and BE the git
    worktree root (not a subdir of one). Returns the resolved root or ``None`` for
    empty input; raises ``WorkspaceRootError``."""
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


def remote_session_facts(
    ref: SshWorkspaceRef,
    *,
    project_id: str = "",
    task_id: str = "",
) -> dict:
    """Open-or-reuse the target's project session and return its admission facts.

    The ONE seam where the ssh branch consults the target, and it consults the
    session broker rather than opening a probe of its own — the broker is already
    the single owner of every remote session for this server generation, so a
    second door would mean a second transport with a second lifetime that panic
    would have to know about.

    Admission is idempotent per session key: the broker reuses an existing session,
    so asking twice (once to verify the worktree, once for the preflight) costs one
    round trip, not two. When no project is named the workspace is its own scope —
    the session key needs A scope, and inventing a shared one would let two
    unrelated placements collide in the broker's session table.

    Every transport-level failure becomes :class:`RemoteWorkspaceUnavailableError`:
    at admission the only alternative to refusing is running the task on the wrong
    host.
    """
    return _session_facts(
        ref.connection_id,
        ref.remote_root,
        workspace_id=ref.workspace_id,
        project_id=str(project_id or "").strip() or ref.workspace_id,
        task_id=task_id,
    )


def _session_facts(
    connection_id: str,
    remote_root: str,
    *,
    workspace_id: str,
    project_id: str,
    task_id: str = "",
) -> dict:
    """:func:`remote_session_facts` over the three identity strings.

    Parameterized because a placement being CREATED has no workspace identity yet —
    that identity belongs to the target (see :func:`admit_remote_placement`) — while
    a placement being re-verified has one and must be checked against it. One body,
    both callers; a second copy is how the two would start asking differently.
    """
    from ouroboros.connection_store import get_connection
    from ouroboros.remote_workspace import get_remote_workspace_service
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    row = get_connection(connection_id, include_retired=False)
    if row is None:
        raise WorkspaceRootError(
            f"workspace_ref.connection_id is not an active remote connection: {connection_id}"
        )
    try:
        service = get_remote_workspace_service()
        admit = getattr(service, "admit_workspace", None)
        if not callable(admit):
            raise RemoteWorkspaceUnavailableError(
                f"connection {connection_id!r} cannot be admitted: the broker in this "
                "process exposes no session admission",
                # A broker that is running but cannot admit is a Home-side wiring
                # state, not something the target or a Bootstrap can move.
                action=REFUSAL_ACTIONS["remote_workspace_unavailable"],
            )
        return dict(
            admit(
                row,
                remote_root=remote_root,
                project_id=project_id,
                workspace_id=workspace_id,
                task_id=str(task_id or ""),
            )
        )
    except RemoteWorkspaceError as exc:
        # `cause=exc`, not a bare message: the broker's refusal already knows what
        # removes it, and flattening it to a string is what made both HTTP callers
        # invent `bootstrap_connection` for every reason a target can refuse.
        raise RemoteWorkspaceUnavailableError(
            f"ssh workspace {remote_root} on connection {connection_id!r} cannot be "
            f"verified on the target ({exc.code}: {exc})",
            cause=exc,
        ) from exc


def _require_known_connection(connection_id: str) -> None:
    """The identity check Home alone CAN decide: the owner's store must know this id.

    Raises :class:`RemoteWorkspaceUnavailableError` when there is no store to ask
    (an unverifiable connection identity is not an admissible one) and
    ``WorkspaceRootError`` when the store simply does not know the id.
    """
    trusted = known_connections()
    if trusted is None:
        raise RemoteWorkspaceUnavailableError(
            f"connection {connection_id!r} cannot be verified: no remote connection store in this build",
            action=REFUSAL_ACTIONS["connection_store_unavailable"],
        )
    if connection_id not in trusted:
        raise WorkspaceRootError(
            f"workspace_ref.connection_id is not a known remote connection: {connection_id}"
        )


def _require_target_worktree_root(facts: dict, remote_root: str) -> None:
    """The remote twin of the local ``git rev-parse --show-toplevel`` gate.

    execd admits only at a canonical git toplevel, so a ref naming a subdirectory of
    a worktree is refused WITH the toplevel it should have named instead. Judged in
    the target's POSIX spellings; no Home ``pathlib`` touches either side.
    """
    toplevel = str(facts.get("canonical_root") or "")
    if not toplevel:
        raise WorkspaceRootError(
            f"workspace_ref.remote_root must be a git worktree root on the target: {remote_root}"
        )
    if toplevel != remote_root:
        raise WorkspaceRootError(
            f"workspace_ref.remote_root must be the git worktree root: {toplevel}"
        )


def admit_remote_placement(
    *,
    connection_id: Any,
    remote_root: Any,
    project_id: str,
) -> SshWorkspaceRef:
    """SEAL a project's remote placement from the owner's HALF-OPEN request.

    The owner names the two halves they can know: WHICH host (a connection their
    store already trusts) and WHICH path on it. The third field of a placement — the
    workspace identity — is deliberately NOT theirs to name: it is allocated by the
    target when the worktree is first attached, and a client that could name it could
    claim a workspace it never opened. So this door takes ``connection_id`` +
    ``remote_root``, consults the target through the same broker session admission
    every other remote placement uses, and returns the sealed three-field ref.

    This is why the project surface does not accept a serialized ``workspace_ref``
    blob: a client choosing the discriminated variant directly would be choosing its
    own placement authority, and the only variant it could legitimately fill is the
    one whose identity field it cannot know.

    Raises ``WorkspaceRootError`` for a malformed request or a path that is not the
    target's canonical worktree root, and :class:`RemoteWorkspaceUnavailableError`
    when the target cannot be consulted at all.
    """
    try:
        clean_connection = str(connection_id or "").strip()
        if not clean_connection or any(
            ord(char) < 32 or ord(char) == 127 for char in clean_connection
        ):
            raise ValueError("workspace_ref.connection_id is required")
        root = normalize_remote_root(remote_root)
    except ValueError as exc:
        raise WorkspaceRootError(str(exc)) from exc
    pid = str(project_id or "").strip()
    if not pid:
        raise WorkspaceRootError(
            "a remote placement is admitted for a NAMED project: the project id is the "
            "broker's session scope and the registry row that carries its routing generation"
        )
    _require_known_connection(clean_connection)
    facts = _session_facts(clean_connection, root, workspace_id="", project_id=pid)
    _require_target_worktree_root(facts, root)
    admitted = str(facts.get("workspace_id") or "")
    if not admitted:
        raise RemoteWorkspaceUnavailableError(
            f"the target returned no workspace identity for {root} on connection "
            f"{clean_connection!r}; the placement cannot be sealed",
            # An admission that answers without allocating an identity is an
            # executor that does not implement this contract set: Bootstrap
            # installs the one that does.
            action=REFUSAL_ACTIONS["remote_execd_outdated"],
        )
    return SshWorkspaceRef(
        connection_id=clean_connection, remote_root=root, workspace_id=admitted
    )


def _admit_ssh_workspace_ref(
    ref: SshWorkspaceRef,
    *,
    drive_root: Any,
    project_id: str = "",
) -> SshWorkspaceRef:
    """The SSH admission branch: full local FORM validation, then target facts.

    The form is already normalized by ``normalize_workspace_ref`` (absolute
    normalized POSIX root, no traversal segments, no control characters, non-empty
    connection/workspace identity, no unknown fields). What is added here is the
    identity check Home alone CAN decide — the connection must be one the owner's
    store knows — and the target-fact gate, which must never be answered by Home.

    The canonical-root gate is the remote twin of the local
    ``git rev-parse --show-toplevel`` check and is shared with
    :func:`admit_remote_placement`, which admits the same placement for the FIRST
    time; the difference between the two branches is only that this one already
    knows the workspace identity and therefore verifies it instead of accepting it.
    """
    _require_known_connection(ref.connection_id)
    facts = remote_session_facts(ref, project_id=project_id)
    _require_target_worktree_root(facts, ref.remote_root)
    admitted = str(facts.get("workspace_id") or "")
    if admitted and admitted != ref.workspace_id:
        raise WorkspaceRootError(
            "workspace_ref.workspace_id no longer identifies the admitted worktree: "
            f"{admitted}"
        )
    return ref


def validate_workspace_root(
    value: Any,
    *,
    system_repo_dir: Any,
    drive_root: Any,
    project_id: str = "",
) -> Optional[WorkspaceRef]:
    """THE admission SSOT: validate a REQUESTED placement and return it SEALED.

    Polymorphic over what the caller asked for (RWS v2 §3.3):

    * a path-ish string (or ``None``/empty) → the local branch,
      ``_admit_local_workspace_root``, byte-identical to v6.58.0;
    * a serialized/typed ``WorkspaceRef`` → its own discriminated branch.

    Returns ``None`` for "no workspace requested", otherwise a sealed
    ``WorkspaceRef``. Raises ``WorkspaceRootError`` (a ``ValueError``) for a
    malformed/unusable request and ``RemoteWorkspaceUnavailableError`` when an
    ssh placement is well-formed but its target cannot be consulted. There is no
    third outcome: an unverifiable placement never becomes a Home path.
    """
    if isinstance(value, (LocalWorkspaceRef, SshWorkspaceRef)) or isinstance(value, dict):
        try:
            requested = normalize_workspace_ref(value)
        except ValueError as exc:
            raise WorkspaceRootError(str(exc)) from exc
        if requested is None:
            return None
        if isinstance(requested, SshWorkspaceRef):
            return _admit_ssh_workspace_ref(
                requested, drive_root=drive_root, project_id=project_id
            )
        value = requested.local_root
    root = _admit_local_workspace_root(value, system_repo_dir=system_repo_dir, drive_root=drive_root)
    return LocalWorkspaceRef(local_root=str(root)) if root is not None else None


def local_admitted_path(ref: Any) -> Optional[pathlib.Path]:
    """Home path of an ADMITTED placement, or ``None`` when there is no workspace.

    The single projection admission callers use when they need a Home ``Path``:
    an ssh ref raises ``RemoteWorkspacePathError`` from ``home_path()`` — the
    typed refusal fires at the ref seam, before any consumer can ``.resolve()``
    a remote spelling (Appendix C-2, P1).
    """
    return None if ref is None else ref.home_path()


# Sentinel: an explicit "no workspace" for a task born in a project room — distinct
# from an unset/empty value (which means "use the room's working_dir by default").
WORKSPACE_NONE = "none"


def resolve_room_workspace(
    *,
    drive_root: Any,
    system_repo_dir: Any,
    project_id: str,
    explicit_workspace: Any = "",
    workspace_sentinel: str = "",
) -> tuple[Optional[WorkspaceRef], str]:
    """Resolve the SEALED placement for a task born in a project room (promote/route).

    Precedence (P5 — the semantic "this work belongs to project X" is already the
    LLM's/owner's decision; this only supplies the room's placement as transport):

    - ``workspace_sentinel == "none"`` → no workspace (explicit opt-out), returns (None,"").
    - an ``explicit_workspace`` the caller passed (path string OR serialized ref)
      → validated and used as-is.
    - else the project's registered placement: its serialized ``placement`` ref if
      the project is REMOTE, else its ``working_dir`` → validated and used.
    - else no workspace (a file-less project), returns (None,"").

    Returns ``(ref, error)``. ``error`` is non-empty when a workspace was REQUESTED
    (explicit placement or a set project placement) but is unusable, AND when the
    project's registry entry cannot be READ at all — the caller MUST fail the task
    loudly rather than fall back to a workspace-less self_modification profile (the
    loud-fail invariant). A REMOTE project room is the C-2:102 resolution: the
    orchestration stays Home, the VALIDATION is the ssh branch's, and an unverifiable
    remote room yields the loud error — never a Home ``working_dir`` stand-in."""
    if str(workspace_sentinel or "").strip().lower() == WORKSPACE_NONE:
        return None, ""

    requested: Any = explicit_workspace if isinstance(explicit_workspace, dict) else str(explicit_workspace or "").strip()
    source = "explicit workspace_root"
    if not requested and str(project_id or "").strip():
        try:
            from ouroboros.projects_registry import get_project

            project = get_project(drive_root, project_id) or {}
            placement = project.get("placement")
            # A remote project stores its placement ref and NO working_dir: there is
            # no Home folder to fall back to, by construction.
            requested = (
                placement
                if isinstance(placement, dict) and placement
                else str(project.get("working_dir") or "").strip()
            )
            source = f"project {project_id!r} working_dir"
        except Exception as exc:
            # BIND-OR-LOUD-FAIL (upstream). "The registry could not be read" is NOT
            # the same fact as "this project has no placement", and collapsing the
            # two silently ran the task workspace-less on the self_modification
            # profile over the SYSTEM repo. An unreadable registry is an error; the
            # caller fails the task loudly. Kept over v2's swallowing variant.
            log.warning(
                "resolve_room_workspace: project registry read failed for %r", project_id, exc_info=True
            )
            return None, (
                f"project {project_id!r} registry entry is unreadable "
                f"({type(exc).__name__}: {exc}) — cannot determine the task's workspace"
            )
    if not requested:
        return None, ""  # file-less project (or no working_dir): a non-workspace task

    try:
        resolved = validate_workspace_root(
            requested, system_repo_dir=system_repo_dir, drive_root=drive_root
        )
    except WorkspaceRootError as exc:
        # LOUD FAIL: a set-but-broken working_dir must never silently degrade to a
        # workspace-less (self_modification-profile) task over the system repo.
        return None, f"{source} is unusable: {exc}"
    return resolved, ""


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


# Task-level fence the non-gateway creation surfaces seal alongside the ref, and the
# queue revalidates under `_queue_lock` immediately before appending to PENDING
# (RWSB2-05). Task-level rather than metadata by design: it is queue-admission
# bookkeeping, it must never ride into `ToolContext.task_metadata`, and it must never be
# inheritable from a schedule template.
PLACEMENT_FENCE_KEY = "_placement_fence"


def placement_fence_for(drive_root: Any, project_id: str, ref: Any) -> dict:
    """Bind a resolved placement to the routing facts that were TRUE when it resolved.

    A schedule fires, the placement resolves, and only then is the task inserted into
    PENDING — a window in which the owner can delete the project, rebind it elsewhere, or
    retire the connection. The fence records the project's ``routing_generation`` and the
    connection's trust identity so the queue can tell "still the same routing" from
    "resolved against a world that no longer exists", instead of inserting a stale
    placement and discovering it at execution time on the wrong host.
    """
    pid = str(project_id or "").strip()
    fence = {"project_id": pid, "routing_generation": -1, "connection_id": "", "connection_trust": ""}
    if pid:
        try:
            from ouroboros.projects_registry import get_reserved_project

            project = get_reserved_project(drive_root, pid) or {}
            fence["routing_generation"] = int(project.get("routing_generation") or 0)
        except Exception:
            fence["routing_generation"] = -1
    if isinstance(ref, SshWorkspaceRef):
        fence["connection_id"] = ref.connection_id
        fence["connection_trust"] = (known_connections() or {}).get(ref.connection_id, "")
    return fence


def placement_fence_stale_reason(drive_root: Any, fence: Any) -> str:
    """"" when the fenced routing still holds, else a typed refusal reason code.

    Called under the queue lock immediately before insertion. A stale generation is a
    REFUSAL, not a re-resolution: silently re-resolving would mean the owner's rebind
    retroactively moved work they scheduled against the previous target.
    """
    if not isinstance(fence, dict):
        return ""
    pid = str(fence.get("project_id") or "").strip()
    if pid:
        try:
            from ouroboros.projects_registry import get_reserved_project

            project = get_reserved_project(drive_root, pid)
        except Exception:
            return "placement_project_lookup_failed"
        if project is None:
            return "placement_project_missing"
        # Explicit int coercion, never `or`: generation 0 is the FIRST generation of every
        # project, and a falsy-default would read it as "unknown" and refuse every task.
        try:
            fenced_generation = int(fence.get("routing_generation"))
        except (TypeError, ValueError):
            fenced_generation = -1
        if int(project.get("routing_generation") or 0) != fenced_generation:
            return "placement_routing_generation_stale"
    connection_id = str(fence.get("connection_id") or "").strip()
    if connection_id:
        trusted = known_connections()
        if trusted is None or connection_id not in trusted:
            return "placement_connection_retired"
        if trusted.get(connection_id, "") != str(fence.get("connection_trust") or ""):
            return "placement_connection_trust_changed"
    return ""


def placement_preflight_summary(
    ref: Any,
    *,
    timeout_sec: float = 8.0,
    project_id: str = "",
) -> dict:
    """Preflight for an ADMITTED placement, in the placement's own terms.

    Local: the existing hard-capped Home snapshot. Remote: the target's own session
    admission facts, because a Home walk over a tree that lives on another host is
    not slow — it answers about the wrong filesystem.

    THE preflight door for both admission surfaces. When the gateway kept its own
    copy of the remote arm, the two drifted immediately; one of them is enough.

    A remote preflight that cannot be taken DISCLOSES that instead of raising: by
    the time this runs the placement is already admitted, so the honest answer is a
    summary that says the facts are missing — a task must not be lost because its
    context block could not be decorated.
    """
    if ref is None:
        return {}
    from ouroboros.workspace_ref import workspace_display_root

    display_root = workspace_display_root(ref)
    if not isinstance(ref, SshWorkspaceRef):
        return bounded_workspace_preflight(display_root, timeout_sec=timeout_sec)
    try:
        facts = remote_session_facts(ref, project_id=project_id)
    except Exception as exc:
        return {
            "schema_version": 1,
            "workspace_root": display_root,
            "error": f"{getattr(exc, 'code', type(exc).__name__)}: {exc}",
        }
    handshake = facts.get("handshake")
    handshake = dict(handshake) if isinstance(handshake, dict) else {}
    platform = handshake.get("platform")
    platform = dict(platform) if isinstance(platform, dict) else {}
    return {
        "schema_version": 1,
        "workspace_root": display_root,
        # Target IDENTITY, not a directory walk: what the owner and the model need
        # to know is which host and which build answered, and that the placement is
        # the worktree root it claimed to be. The tree itself is read with the
        # workspace tools, which run natively there — a listing taken here would be
        # a second, staler answer to a question those tools already answer.
        "placement": "ssh",
        "connection_id": ref.connection_id,
        "host_id": str(facts.get("host_id") or ""),
        "canonical_root": str(facts.get("canonical_root") or ""),
        "release_id": str(facts.get("release_id") or ""),
        "target_system": str(platform.get("system") or ""),
        "target_python": str(platform.get("python") or ""),
    }


def seal_admitted_placement(task: dict, ref: Any, *, preflight_summary: dict) -> None:
    """Write an ADMITTED placement into a task dict — ONE composition, two callers.

    The promote path and the schedule-fire path both turn a resolved placement into a
    workspace task, and when they each did it themselves the promote path drifted into a
    degraded twin that skipped validation entirely. So the sealing (ref + display
    spelling + external/forked modes + the [HEADLESS_WORKSPACE] block) lives here once.
    ``/api/tasks`` composes its own task dict from the request but seals the SAME key
    with the same ref, which is what the sealing tests pin.
    """
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY, workspace_display_root

    display_root = workspace_display_root(ref)
    if not display_root:
        return
    task["workspace_root"] = display_root
    task["workspace_mode"] = "external"
    task["memory_mode"] = "forked"
    metadata = task.setdefault("metadata", {})
    metadata[SEALED_WORKSPACE_REF_KEY] = ref.to_payload()
    metadata["workspace_root"] = display_root
    metadata["workspace_preflight"] = preflight_summary
    task["text"] = (
        f"{task.get('text') or ''}\n\n[HEADLESS_WORKSPACE]\n"
        + compose_workspace_block(
            workspace_root=display_root,
            workspace_mode="external",
            memory_mode="forked",
            workspace_preflight=preflight_summary,
        )
        + "[END_HEADLESS_WORKSPACE]"
    )


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
