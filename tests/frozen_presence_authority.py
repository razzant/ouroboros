"""Construct historical ceilings for compatibility tests; production only decodes them."""
from ouroboros.presence_authority import (
    PresenceAuthorityError, PresenceArgumentBinding, PresenceToolGrant,
    PresenceResourceGrant, PresenceCapabilityCeiling, presence_ceiling_payload, _text, _sha,
    PRESENCE_OWN_WORK_SCOPE, _SHA256_LEN,
)
from tests.frozen_presence_capabilities import (
    PresenceProfileResolution, PresenceToolTarget, PresenceScriptTarget, PresenceResourceTarget,
)
from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES
_OWN_WORK_BASELINE = (("get_task_result", True), ("recent_tasks", True), ("tell_task", False))

def build_presence_capability_ceiling(
    *,
    skill_name: str,
    skill_content_hash: str,
    state_fingerprint: str,
    resolution: PresenceProfileResolution,
) -> PresenceCapabilityCeiling:
    """Compile active selections; missing required requests refuse admission."""

    if not isinstance(resolution, PresenceProfileResolution) or not resolution.required_selections_present:
        raise PresenceAuthorityError("presence_authority_missing_required", "resolution.missing_required")
    tools: list[PresenceToolGrant] = []
    resources: list[PresenceResourceGrant] = []
    for selection in resolution.active:
        target = selection.target
        if isinstance(target, PresenceToolTarget):
            tools.append(PresenceToolGrant(target.name, selection.argument_bindings))
        elif isinstance(target, PresenceScriptTarget):
            bindings = (
                PresenceArgumentBinding(("skill",), "static", static_value=target.skill),
                PresenceArgumentBinding(("script",), "static", static_value=target.script),
                *selection.argument_bindings,
            )
            tools.append(PresenceToolGrant("skill_exec", tuple(bindings)))
        elif isinstance(target, PresenceResourceTarget):
            resources.append(
                PresenceResourceGrant(
                    target.root,
                    target.operations,
                    target.path_prefix,
                    target.bucket,
                    target.skill_name,
                )
            )
    # The cognitive baseline is part of the ceiling, not of the profile: an
    # admitted conversation is still this mind, so it keeps its own memory in
    # every room instead of losing what the exchange taught it. It adds no
    # authority over settings, delivery, or the filesystem. A name the profile
    # already selected keeps THAT grant, because its host-authored argument
    # bindings are the exact facts apply_presence_argument_bindings overrides
    # the model with; an unselected name arrives with no bindings at all.
    selected = {grant.name for grant in tools}
    tools.extend(
        PresenceToolGrant(name)
        for name in sorted(COGNITIVE_MEMORY_TOOL_NAMES)
        if name not in selected
    )
    scope = (PresenceArgumentBinding(("presence_scope",), "static", static_value=PRESENCE_OWN_WORK_SCOPE),)
    tools.extend(
        PresenceToolGrant(name, scope if scoped else ())
        for name, scoped in _OWN_WORK_BASELINE
        if name not in selected
    )
    provisional = PresenceCapabilityCeiling(
        skill_name=_text(skill_name, "skill_name"),
        skill_content_hash=_sha(skill_content_hash, "skill_content_hash"),
        profile_fingerprint=resolution.profile_fingerprint,
        state_fingerprint=_sha(state_fingerprint, "state_fingerprint"),
        selection_fingerprint=resolution.selection_fingerprint,
        model_slot=resolution.runtime.model_slot,
        inline_max_rounds=resolution.runtime.inline_max_rounds,
        tool_grants=tuple(sorted(tools, key=lambda item: item.name)),
        resource_grants=tuple(sorted(resources, key=lambda item: (item.root, item.path_prefix))),
        digest="0" * _SHA256_LEN,
    )
    payload = presence_ceiling_payload(provisional)
    return PresenceCapabilityCeiling(
        **{**provisional.__dict__, "digest": payload["digest"]}
    )
