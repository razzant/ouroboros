"""Who is acting and where each resource root physically lives.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
"""

from __future__ import annotations

import pathlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from ouroboros.tool_access_types import Operation
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tool_access_types import ResourceRoot
    from ouroboros.tool_access_types import ToolProfile
    from typing import Any
    from typing import Optional


def _tool_access():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros import tool_access

    return tool_access


def _is_subagent_ctx(ctx: Any) -> bool:
    """True when the task is a delegated subagent (by lineage metadata)."""
    for attr in ("task_metadata", "task_contract"):
        data = getattr(ctx, attr, None)
        if isinstance(data, dict) and str(data.get("delegation_role") or "").strip() == "subagent":
            return True
    return False


def is_external_workspace(ctx: Any) -> bool:
    """True for an EXTERNAL-workspace top-level task (not the system repo).

    External-workspace tasks operate on a pre-existing working tree somewhere on
    the host (container scratch, a repo cloned under ``/tmp`` or ``/build``,
    etc.). They legitimately read, run commands, and use git OUTSIDE the user
    home, while the Ouroboros runtime (system repo + data drive) and
    credential-like files stay protected by the per-path guards. ``self_worktree``
    and ``genesis`` are acting-subagent SURFACES (``acting_subagent`` profile),
    never this profile, so they keep full home/runtime confinement.
    """
    try:
        if not bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
            return False
    except Exception:
        return False
    return str(getattr(ctx, "workspace_mode", "") or "").strip().lower() == "external"


def active_tool_profile(ctx: Any) -> ToolProfile:
    metadata = getattr(ctx, "task_metadata", None) or {}
    frozen = metadata.get("tool_profile")
    if frozen == "local_readonly_subagent":
        return "local_readonly_subagent"
    constraint = _tool_access().normalize_task_constraint(getattr(ctx, "task_constraint", None))
    mode = str(getattr(constraint, "mode", "") or "").strip()
    if mode == _tool_access().LOCAL_READONLY_SUBAGENT_MODE:
        return "local_readonly_subagent"
    if mode == _tool_access().ACTING_SUBAGENT_MODE:
        # Acting subagents require a resolved write surface; otherwise fail
        # closed to read-only rather than inheriting a broader profile.
        surface = str(getattr(constraint, "surface", "") or "").strip()
        if surface in _tool_access().VALID_WRITE_SURFACES:
            return "acting_subagent"
        return "local_readonly_subagent"
    # Fail-closed floor (BIBLE P3), checked BEFORE workspace/direct-chat: a
    # delegated subagent without a valid readonly/acting constraint is
    # read-only and must never inherit workspace_task / operator_control /
    # self_modification. The parent remains the sole local writer/committer.
    if _is_subagent_ctx(ctx):
        return "local_readonly_subagent"
    if frozen in {"operator_control", "self_modification", "workspace_task",
                  "external_workspace_task", "local_readonly_subagent", "acting_subagent"}:
        return frozen
    if bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        # Keep distinct preset names for focus/path diagnostics. Both use the
        # shared ordinary principal; external host-scratch reach is a path fact.
        if is_external_workspace(ctx):
            return "external_workspace_task"
        return "workspace_task"
    if bool(getattr(ctx, "is_direct_chat", False)):
        return "operator_control"
    return "self_modification"


def freeze_tool_profile(task: dict) -> str:
    """Bind admission authority once; moving execution never selects a new preset.

    Delegated constraints still take precedence in active_tool_profile. Presence
    and consciousness keep their separate binding/disabled-tool ceilings.
    """
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        task_constraint=task.get("task_constraint"), task_contract=task.get("task_contract"),
        task_metadata={**{key: value for key, value in (task.get("metadata") or {}).items() if key != "tool_profile"},
                       **{key: task[key] for key in ("tool_profile", "delegation_role") if key in task}},
        is_workspace_mode=lambda: bool(task.get("workspace_root") and str(task.get("workspace_mode") or "").strip()),
        workspace_mode=task.get("workspace_mode"), is_direct_chat=bool(task.get("_is_direct_chat")))
    task["tool_profile"] = active_tool_profile(ctx)
    return task["tool_profile"]


def predicted_subagent_profile(*, write_surface: str = "") -> ToolProfile:
    """The tool profile a scheduled subagent will resolve to, from schedule-time
    inputs only (v6.57.0, 1.6). A valid write_surface → acting_subagent; otherwise
    a read-only subagent. Mirrors active_tool_profile's subagent branches so the
    parent's schedule result and the child's start context can preview the profile
    without a live ctx. NOT authoritative — the supervisor's _resolve_subagent_
    constraint is the real gate; this is a visibility preview."""
    surface = str(write_surface or "").strip()
    if surface and surface in _tool_access().VALID_WRITE_SURFACES:
        return "acting_subagent"
    return "local_readonly_subagent"


def project_room_lens_dir(ctx: Any) -> Optional[pathlib.Path]:
    """Return a direct-chat room's selected folder, otherwise ``None``.

    Promoted/workspace/subagent tasks carry their own workspace; only a direct
    chat without one may use the host's ``_project_room_dir``. Its address stays
    selected if the folder disappears: existence is an operation's concern,
    never permission to switch back to the system repository.
    """
    if not bool(getattr(ctx, "is_direct_chat", False)):
        return None
    if getattr(ctx, "workspace_root", None):
        return None
    meta = getattr(ctx, "task_metadata", None)
    raw = str(meta.get("_project_room_dir") or "").strip() if isinstance(meta, dict) else ""
    if not raw:
        note = str(meta.get("_project_room_note") or "") if isinstance(meta, dict) else ""
        if note:
            raise ValueError(note)
        return None
    return pathlib.Path(raw).resolve(strict=False)


def folderless_scratch_dir(ctx: Any) -> Optional[pathlib.Path]:
    """Project work WITHOUT a folder: the task's OWN scratch is its default cwd (#1315).

    Project-scoped work with no workspace and no room folder (or an explicit
    "no folder" intent) no longer defaults to the Ouroboros repository. This is a
    default location only — never a workspace: no external-workspace or delegated
    write authority, no change to what the profile may reach (the system repo stays
    one explicit ``root``/``cwd`` away). Main self-work, Presence and an explicit
    ``system_repo`` intent keep today's default."""
    if getattr(ctx, "workspace_root", None):
        return None
    meta = getattr(ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    intent = meta.get("resource_intent") if isinstance(meta.get("resource_intent"), dict) else {}
    kind = str(intent.get("kind") or "")
    project = str(getattr(ctx, "project_id", "") or meta.get("project_id") or "").strip()
    if (kind == "system_repo" or (not project and kind != "explicit_none")
            or meta.get("_project_room_dir") or meta.get("_project_room_note")):
        return None
    from ouroboros.dialogue_provenance import presence_metadata_binding

    if presence_metadata_binding(meta) is not None:
        return None
    scratch = resource_root_path(ctx, "task_drive")
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def load_bound_skill(binding: ResolvedResourceBinding) -> Any:
    """Load the frozen payload target while preserving lifecycle provenance."""
    from ouroboros.skill_loader import _classify_skill_source, load_skill
    loaded = load_skill(binding.base_path, binding.state_drive_root)
    if loaded is not None:
        loaded.source = _classify_skill_source(
            binding.base_path,
            location=binding.source,
            drive_root=binding.state_drive_root,
        )
    return loaded


def _skill_payload_base(
    ctx: Any,
    *,
    profile: ToolProfile,
    operation: Operation,
    location: str,
    skill_name: str,
    allow_missing: bool = False,
) -> tuple[pathlib.Path, str, str]:
    """Select one physical package and project its effective source."""
    from ouroboros.skill_payload_binding import resolve_skill_payload_base

    return resolve_skill_payload_base(
        ctx,
        drive_root=_tool_access().canonical_data_root(ctx),
        profile=profile,
        top_level=profile in _tool_access()._TOP_LEVEL_PRINCIPAL_PROFILES,
        operation=operation,
        location=location,
        skill_name=skill_name,
        allow_missing=allow_missing,
    )


def resource_root_path(
    ctx: Any,
    root: ResourceRoot,
    *,
    bucket: str = "",
    skill_name: str = "",
) -> pathlib.Path:
    if root == "active_workspace":
        active = getattr(ctx, "active_repo_dir", None)
        candidate = active() if callable(active) else (project_room_lens_dir(ctx) or folderless_scratch_dir(ctx))
        if candidate is None or candidate.__class__.__module__.startswith("unittest.mock"):
            candidate = getattr(ctx, "repo_dir")
        return pathlib.Path(candidate).resolve(strict=False)
    if root == "system_repo":
        return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")).resolve(strict=False)
    if root == "runtime_data":
        return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)
    if root == "task_drive":
        return (pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False) / "task_drives" / _tool_access().task_id_for_artifacts(ctx)).resolve(strict=False)
    if root == "artifact_store":
        return _tool_access().task_artifact_dir_path(pathlib.Path(getattr(ctx, "drive_root")), _tool_access().task_id_for_artifacts(ctx), create=False).resolve(strict=False)
    if root == "user_files":
        return _tool_access()._user_files_root()
    if root == "subagent_projects":
        from ouroboros.config import get_subagent_projects_root

        return pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
    if root == "deliverables":
        return _tool_access()._deliverables_root()
    if root == "skill_payload":
        b = str(bucket or "").strip()
        s = str(skill_name or "").strip()
        if not b or not s:
            raise ValueError("root=skill_payload requires bucket and skill_name")
        base, _source, _name = _skill_payload_base(
            ctx,
            profile=active_tool_profile(ctx),
            operation="read",
            location=b,
            skill_name=s,
        )
        return base
    raise ValueError(f"unknown root {root!r}")


def binding_targets_system_repo(
    ctx: Any, binding: ResolvedResourceBinding | None,
) -> bool:
    """Whether a selected logical root physically lands on Ouroboros source."""

    if binding is None:
        return False
    try:
        return pathlib.Path(binding.base_path).resolve(strict=False) == resource_root_path(
            ctx, "system_repo",
        )
    except (OSError, TypeError, ValueError):
        return False
