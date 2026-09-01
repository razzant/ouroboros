from __future__ import annotations

import pathlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.skill_review_status import skill_review_gate

log = logging.getLogger(__name__)


@dataclass
class SkillReadiness:
    ready: bool
    blockers: List[str] = field(default_factory=list)
    agent_fixable_blockers: List[str] = field(default_factory=list)
    owner_action_blockers: List[str] = field(default_factory=list)
    review_gate: Dict[str, Any] = field(default_factory=dict)
    grant_status: Dict[str, Any] = field(default_factory=dict)
    conflict: Dict[str, Any] = field(default_factory=dict)
    # G3 (capinv-447): declared dependencies that need manual installation —
    # disclosed ("kind:package"), never silently dropped; not a hard blocker.
    manual_dependencies: List[str] = field(default_factory=list)


def skill_readiness_for_execution(
    drive_root: pathlib.Path,
    skill: Any,
    *,
    require_enabled: bool = True,
    require_grants: bool = True,
    skills: Optional[List[Any]] = None,
) -> SkillReadiness:
    blockers: List[str] = []
    agent_fixable: List[str] = []
    owner_action: List[str] = []

    if getattr(skill, "load_error", ""):
        msg = f"load_error={skill.load_error!r}"
        blockers.append(msg)
        agent_fixable.append(msg)

    stale = skill.review.is_stale_for(skill.content_hash)
    gate = skill_review_gate(skill.review.status, stale=stale)
    if stale:
        blockers.append("review_stale")
        agent_fixable.append("review_stale")
    elif not gate.get("executable_review"):
        reason = str(gate.get("blocking_reason") or "review_not_executable")
        msg = f"review_not_executable:{reason}"
        blockers.append(msg)
        agent_fixable.append(msg)

    if require_enabled and not getattr(skill, "enabled", False):
        blockers.append("skill_disabled")
        owner_action.append("skill_disabled")

    from ouroboros.skill_loader import discover_skills, skill_conflict_status

    peers = skills if skills is not None else discover_skills(pathlib.Path(drive_root))
    conflict = skill_conflict_status(skill, peers) or {}
    if conflict:
        names = list(conflict.get("skills") or [])
        suffix = f":{','.join(names)}" if names else ""
        msg = f"skill_conflict{suffix}"
        blockers.append(msg)
        owner_action.append(msg)

    grants: Dict[str, Any] = {}
    if require_grants:
        from ouroboros.skill_loader import grant_status_for_skill

        grants = grant_status_for_skill(pathlib.Path(drive_root), skill)
        if not grants.get("all_granted", True):
            missing_keys = grants.get("missing_keys") or []
            missing_permissions = grants.get("missing_permissions") or []
            msg = f"missing_grants:keys={missing_keys},permissions={missing_permissions}"
            blockers.append(msg)
            owner_action.append(msg)

    try:
        from ouroboros.marketplace.install_specs import install_specs_hash
        from ouroboros.marketplace.isolated_deps import read_deps_state
        from ouroboros.skill_dependencies import (
            auto_install_specs_for_skill,
            manual_install_specs_for_skill,
        )

        auto_specs = auto_install_specs_for_skill(pathlib.Path(drive_root), skill)
        if auto_specs:
            deps_state = read_deps_state(pathlib.Path(drive_root), skill.name, skill.skill_dir)
            deps_status = str(deps_state.get("status") or "pending")
            if deps_status != "installed":
                msg = f"deps_not_ready:{deps_status}"
                blockers.append(msg)
                agent_fixable.append(msg)
            elif deps_state.get("specs_hash") != install_specs_hash(auto_specs):
                blockers.append("deps_stale")
                agent_fixable.append("deps_stale")
        # G3 (capinv-447) third readiness state: manually-installed dependencies
        # are DISCLOSED, not silently dropped from the dependency list. They do
        # not hard-block (the owner may have installed them system-wide, and no
        # ledger records that), so this is honesty, not a new gate.
        manual_specs, _manual_warnings = manual_install_specs_for_skill(skill)
        manual_dependencies = [
            f"{spec.get('kind') or '?'}:{spec.get('package') or '?'}" for spec in manual_specs
        ]
    except Exception:
        manual_dependencies = []
        log.debug("skill readiness deps probe failed", exc_info=True)

    return SkillReadiness(
        ready=not blockers,
        blockers=blockers,
        agent_fixable_blockers=agent_fixable,
        owner_action_blockers=owner_action,
        review_gate=gate,
        grant_status=grants,
        conflict=conflict,
        manual_dependencies=manual_dependencies,
    )
