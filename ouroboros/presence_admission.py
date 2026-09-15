"""Host-owned admission of authenticated presence turns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ouroboros.presence_authority import (
    PresenceCapabilityCeiling,
    build_presence_capability_ceiling,
)
from ouroboros.presence_bindings import (
    PresenceBindingError,
    PresenceEndpoint,
    load_presence_binding,
)
from ouroboros.presence_capabilities import (
    PresenceScriptTarget,
    PresenceStateError,
    PresenceToolTarget,
    load_presence_state,
    presence_state_fingerprint,
    resolve_presence_profile_state,
)
from ouroboros.presence_profile import PresenceProfileError, parse_presence_profile
from ouroboros.skill_loader import find_skill


class PresenceAdmissionError(ValueError):
    """Typed refusal before an authenticated event reaches the agent loop."""

    def __init__(self, code: str, field: str, message: str = "") -> None:
        self.code = str(code or "presence_admission_failed")
        self.field = str(field or "presence_admission")
        super().__init__(message or f"{self.code}: {self.field}")


@dataclass(frozen=True)
class PresenceAdmission:
    """Immutable reviewed behavior and positive authority for one bound room."""

    binding_id: str
    transport_skill: str
    behavior_skill: str
    origin: PresenceEndpoint
    destination: PresenceEndpoint
    instructions: str
    context_topics: tuple[str, ...]
    model_slot: str
    inline_max_rounds: int
    skill_content_hash: str
    profile_fingerprint: str
    state_fingerprint: str
    selection_fingerprint: str
    capability_ceiling: PresenceCapabilityCeiling
    workspace_root: str = ""


def _component_error(exc: Any) -> PresenceAdmissionError:
    return PresenceAdmissionError(
        str(getattr(exc, "code", "presence_admission_failed")),
        str(getattr(exc, "field", "presence_admission")),
        str(exc),
    )


def _required_selection_ready(root: Path, profile: Any, resolution: Any) -> None:
    """Refuse only required targets that are not live at admission time."""

    from ouroboros.presence_profile import presence_request_fingerprint

    required = {
        presence_request_fingerprint(request): request.request_id
        for request in profile.capability_requests
        if request.required
    }
    builtin_registry = None
    for selection in resolution.active:
        request_id = required.get(selection.request_fingerprint)
        if request_id is None:
            continue
        target = selection.target
        ready = True
        if isinstance(target, PresenceToolTarget) and target.kind == "builtin":
            from ouroboros.tools.registry import ToolRegistry, _builtin_tool_availability

            builtin_registry = builtin_registry or ToolRegistry(
                repo_dir=Path(__file__).resolve().parents[1], drive_root=root,
            )
            entry = builtin_registry._entries.get(target.name)
            ready = bool(entry and _builtin_tool_availability(target.name, builtin_registry._ctx)[0])
        elif isinstance(target, PresenceToolTarget) and target.kind == "extension":
            from ouroboros.extension_loader import get_tool, is_extension_live

            tool = get_tool(target.name)
            ready = bool(
                tool
                and str(tool.get("skill") or "") == target.provider
                and is_extension_live(target.provider, root)
            )
        elif isinstance(target, PresenceToolTarget):
            from ouroboros.mcp_client import ensure_configured_from_settings, get_manager

            ensure_configured_from_settings(refresh=False)
            tool = get_manager().get_tool(target.name)
            ready = bool(tool and str(tool.get("server_id") or "") == target.provider)
        elif isinstance(target, PresenceScriptTarget):
            script_skill = find_skill(root, target.skill)
            scripts = {
                str(item.get("path") or item.get("name") or "")
                for item in (script_skill.manifest.scripts if script_skill else [])
            }
            ready = bool(script_skill and script_skill.available_for_execution and target.script in scripts)
        if not ready:
            raise PresenceAdmissionError(
                "presence_required_capability_unavailable",
                f"capability_requests.{request_id}",
            )


def admit_presence_turn(
    *,
    drive_root: Path,
    authenticated_transport_skill: str,
    binding_id: str,
    global_max_rounds: int,
    repo_path: str | None = None,
) -> PresenceAdmission:
    """Resolve one opaque binding into a frozen, reviewed admission snapshot."""

    root = Path(drive_root)
    try:
        binding = load_presence_binding(
            root,
            authenticated_transport_skill,
            binding_id,
        )
    except PresenceBindingError as exc:
        raise _component_error(exc) from exc

    skill = find_skill(root, binding.behavior_skill, repo_path=repo_path)
    if skill is None:
        raise PresenceAdmissionError(
            "presence_behavior_skill_not_installed",
            "binding.behavior_skill",
        )
    if skill.load_error:
        raise PresenceAdmissionError(
            "presence_behavior_skill_unavailable",
            "binding.behavior_skill",
        )
    if not skill.enabled:
        raise PresenceAdmissionError(
            "presence_behavior_skill_disabled",
            "binding.behavior_skill",
        )
    gate = skill.review.gate_for(skill.content_hash)
    if gate["blocking_reason"] == "review_stale":
        raise PresenceAdmissionError(
            "presence_behavior_review_stale",
            "binding.behavior_skill",
        )
    if not gate["executable_review"]:
        raise PresenceAdmissionError(
            "presence_behavior_review_not_executable",
            "binding.behavior_skill",
        )

    try:
        profile = parse_presence_profile(skill.manifest, skill.skill_dir)
        if profile is None:
            raise PresenceAdmissionError(
                "presence_behavior_profile_missing",
                "binding.behavior_skill",
            )
        state = load_presence_state(root, skill.name)
        from ouroboros.workspace_admission import WorkspaceRootError, validate_workspace_root

        try:
            workspace = validate_workspace_root(
                state.workspace_root,
                system_repo_dir=Path(__file__).resolve().parents[1],
                drive_root=root,
            )
        except WorkspaceRootError as exc:
            raise PresenceAdmissionError(
                "presence_workspace_unusable", "presence_state.workspace_root", str(exc),
            ) from exc
        state_digest = presence_state_fingerprint(state)
        resolution = resolve_presence_profile_state(
            profile,
            state,
            global_max_rounds=global_max_rounds,
        )
        _required_selection_ready(root, profile, resolution)
        ceiling = build_presence_capability_ceiling(
            skill_name=skill.name,
            skill_content_hash=skill.content_hash,
            state_fingerprint=state_digest,
            resolution=resolution,
        )
    except PresenceAdmissionError:
        raise
    except (PresenceProfileError, PresenceStateError, ValueError) as exc:
        raise _component_error(exc) from exc

    return PresenceAdmission(
        binding_id=binding.binding_id,
        transport_skill=binding.transport_skill,
        behavior_skill=skill.name,
        origin=binding.origin,
        destination=binding.destination,
        instructions=profile.instructions,
        context_topics=profile.context_topics,
        model_slot=resolution.runtime.model_slot,
        inline_max_rounds=resolution.runtime.inline_max_rounds,
        skill_content_hash=skill.content_hash,
        profile_fingerprint=resolution.profile_fingerprint,
        state_fingerprint=state_digest,
        selection_fingerprint=resolution.selection_fingerprint,
        capability_ceiling=ceiling,
        workspace_root=str(workspace) if workspace is not None else "",
    )


__all__ = [
    "PresenceAdmission",
    "PresenceAdmissionError",
    "admit_presence_turn",
]
