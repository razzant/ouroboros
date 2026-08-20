"""Who is acting, and where each logical resource root physically lives.

Resolves the acting tool profile from task lineage and constraint (plus its
schedule-time preview), the direct-chat project room lens, the physical path
behind every resource root, the one selected skill package behind
``root=skill_payload``, and the system-repo test a resolved binding answers.
``active_tool_profile`` lives here because ``resource_root_path`` asks it while
selecting a skill payload; ``tool_access`` re-exports both, so every consumer
keeps the same object. The access decision and the binding built from it stay
with ``tool_access``.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES, normalize_task_constraint
from ouroboros.tool_access_types import (
    Operation,
    ResolvedResourceBinding,
    ResourceRoot,
    ToolProfile,
    _TOP_LEVEL_PRINCIPAL_PROFILES,
)
from ouroboros.tool_access_paths import (
    _deliverables_root,
    _user_files_root,
    canonical_data_root,
)


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
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    mode = str(getattr(constraint, "mode", "") or "").strip()
    if mode == LOCAL_READONLY_SUBAGENT_MODE:
        return "local_readonly_subagent"
    if mode == ACTING_SUBAGENT_MODE:
        # Acting subagents require a resolved write surface; otherwise fail
        # closed to read-only rather than inheriting a broader profile.
        surface = str(getattr(constraint, "surface", "") or "").strip()
        if surface in VALID_WRITE_SURFACES:
            return "acting_subagent"
        return "local_readonly_subagent"
    if mode == "skill_repair":
        return "skill_repair"
    # Fail-closed floor (BIBLE P3), checked BEFORE workspace/direct-chat: a
    # delegated subagent without a valid readonly/acting/skill constraint is
    # read-only and must never inherit workspace_task / operator_control /
    # self_modification. The parent remains the sole local writer/committer.
    if _is_subagent_ctx(ctx):
        return "local_readonly_subagent"
    if bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        # Keep distinct preset names for focus/path diagnostics. Both use the
        # shared ordinary principal; external host-scratch reach is a path fact.
        if is_external_workspace(ctx):
            return "external_workspace_task"
        return "workspace_task"
    if bool(getattr(ctx, "is_direct_chat", False)):
        return "operator_control"
    return "self_modification"


def predicted_subagent_profile(*, write_surface: str = "") -> ToolProfile:
    """The tool profile a scheduled subagent will resolve to, from schedule-time
    inputs only (v6.57.0, 1.6). A valid write_surface → acting_subagent; otherwise
    a read-only subagent. Mirrors active_tool_profile's subagent branches so the
    parent's schedule result and the child's start context can preview the profile
    without a live ctx. NOT authoritative — the supervisor's _resolve_subagent_
    constraint is the real gate; this is a visibility preview."""
    surface = str(write_surface or "").strip()
    if surface and surface in VALID_WRITE_SURFACES:
        return "acting_subagent"
    return "local_readonly_subagent"


def project_room_lens_dir(ctx: Any) -> Optional[pathlib.Path]:
    """Return a direct-chat room's verified project cwd, otherwise ``None``.

    Promoted/workspace/subagent tasks carry their own workspace; only a direct
    chat without one may use the injected existing ``_project_room_dir``.
    """
    if not bool(getattr(ctx, "is_direct_chat", False)):
        return None
    if getattr(ctx, "workspace_root", None):
        return None
    meta = getattr(ctx, "task_metadata", None)
    raw = str(meta.get("_project_room_dir") or "").strip() if isinstance(meta, dict) else ""
    if not raw:
        return None
    try:
        candidate = pathlib.Path(raw).resolve(strict=False)
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


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
    """Select one physical skill package without reading lifecycle state."""
    from ouroboros.skill_loader import (
        _sanitize_skill_name,
        _select_skill_location,
        _skill_location_inventory,
    )
    requested_location = str(location or "").strip().lower()
    allowed_locations = {"external", "clawhub", "ouroboroshub", "native", "user_repo"}
    canonical_name = _sanitize_skill_name(skill_name)
    if not str(skill_name or "").strip() or canonical_name == "_unnamed":
        raise ValueError("root=skill_payload requires a non-empty skill_name")
    state_root = canonical_data_root(ctx)
    candidates = _skill_location_inventory(state_root)
    if not requested_location and operation == "review":
        identity = tuple(item for item in candidates if item.name == canonical_name)
        if not identity:
            raise ValueError(f"skill {canonical_name!r} was not found")
        requested_location = identity[0].location
    elif requested_location not in allowed_locations:
        raise ValueError(
            "root=skill_payload requires bucket/location in "
            "external|clawhub|ouroboroshub|native|user_repo"
        )
    if requested_location in {"native", "user_repo"} and profile not in _TOP_LEVEL_PRINCIPAL_PROFILES:
        raise ValueError(
            f"profile={profile} cannot select skill location={requested_location}"
        )
    if requested_location == "native" and operation in {"write", "edit", "shell"}:
        raise ValueError(
            "installed native skills are read/review only; edit their seed via root=system_repo"
        )

    selected = _select_skill_location(
        candidates,
        name=canonical_name,
        location=requested_location,
        require_unique_identity=operation not in {"read", "list", "search"},
    )
    if selected is not None:
        return selected.skill_dir.resolve(strict=False), selected.location, selected.name
    if (
        operation == "write"
        and allow_missing
        and requested_location in {"external", "clawhub", "ouroboroshub"}
    ):
        return (
            (state_root / "skills" / requested_location / canonical_name).resolve(strict=False),
            requested_location,
            canonical_name,
        )
    raise ValueError(
        f"skill {canonical_name!r} was not found in location {requested_location!r}"
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
        candidate = None
        if callable(active):
            try:
                candidate = active()
            except Exception:
                candidate = None
        if candidate is None or candidate.__class__.__module__.startswith("unittest.mock"):
            candidate = getattr(ctx, "repo_dir")
        return pathlib.Path(candidate).resolve(strict=False)
    if root == "system_repo":
        return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")).resolve(strict=False)
    if root == "runtime_data":
        return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)
    if root == "task_drive":
        return (pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False) / "task_drives" / task_id_for_artifacts(ctx)).resolve(strict=False)
    if root == "artifact_store":
        return task_artifact_dir_path(pathlib.Path(getattr(ctx, "drive_root")), task_id_for_artifacts(ctx), create=False).resolve(strict=False)
    if root == "user_files":
        return _user_files_root()
    if root == "subagent_projects":
        from ouroboros.config import get_subagent_projects_root

        return pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
    if root == "deliverables":
        return _deliverables_root()
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
