"""The liveness authority for one extension: what it should be, and what it is.

Every gate an extension passes before it may run — manifest type, load error,
enabled flag, skill conflict, executable review freshness, owner grants,
isolated-dependency readiness — is answered once here, as a projection over the
discovered skill and the live registries. Callers decide what to do about it;
this module never loads or unloads anything.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.extension_registry_state import _extensions, _load_failures, _lock
from ouroboros.skill_loader import (
    LoadedSkill,
    _sanitize_skill_name,
    discover_skills,
    grant_status_for_skill,
    skill_conflict_status,
    skill_review_gate,
)

log = logging.getLogger(__name__)

def _extension_runtime_state(
    skill: LoadedSkill,
    *,
    current_hash: str | None = None,
    drive_root: pathlib.Path | None = None,
    skills: Optional[List[LoadedSkill]] = None,
    repo_path: str | None = None,
) -> Dict[str, Any]:
    """Return the liveness authority for one extension."""
    from ouroboros.config import get_runtime_mode

    hash_now = current_hash or skill.content_hash
    skill_dir_now = str(skill.skill_dir.resolve())
    review_stale = skill.review.is_stale_for(hash_now)
    with _lock:
        live_bundle = _extensions.get(skill.name)
        live_loaded = bool(
            live_bundle
            and live_bundle.content_hash == hash_now
            and live_bundle.skill_dir == skill_dir_now
        )
        loaded_present = live_bundle is not None
        load_failure = _load_failures.get(skill.name)
        matched_failure = bool(
            load_failure
            and load_failure.content_hash == hash_now
            and load_failure.skill_dir == skill_dir_now
        )

    review_gate = skill_review_gate(skill.review.status, stale=review_stale)
    if drive_root is None:
        drive_root = pathlib.Path(skill.skill_dir).parent.parent.parent
    peers = list(skills) if skills is not None else discover_skills(
        pathlib.Path(drive_root), repo_path=repo_path
    )
    if not any(peer.name == skill.name for peer in peers):
        peers.append(skill)
    conflict = skill_conflict_status(skill, peers)
    grant_status = grant_status_for_skill(pathlib.Path(drive_root), skill)
    grants_usable = bool(grant_status.get("usable", True))
    reason = "ready"
    desired_live = True
    if not skill.manifest.is_extension():
        desired_live = False
        reason = "not_extension"
    elif skill.load_error:
        desired_live = False
        reason = "load_error"
    elif not skill.enabled:
        desired_live = False
        reason = "disabled"
    elif conflict:
        desired_live = False
        reason = "skill_conflict"
    elif not review_gate["executable_review"]:
        desired_live = False
        reason = review_gate["blocking_reason"]
    elif not grants_usable:
        desired_live = False
        reason = "missing_grants"
    # Light mode allows reviewed skills; it only gates repo mutation/escalation.
    elif matched_failure:
        reason = "load_error"

    return {
        "skill": skill.name,
        "type": skill.manifest.type,
        "runtime_mode": get_runtime_mode(),
        "enabled": skill.enabled,
        "review_status": skill.review.status,
        "review_stale": review_stale,
        "review_gate": review_gate,
        "executable_review": review_gate["executable_review"],
        "grant_status": grant_status,
        "conflict": conflict,
        "load_error": skill.load_error or (load_failure.error if matched_failure and load_failure else None),
        "desired_live": desired_live,
        "live_loaded": live_loaded,
        "loaded_present": loaded_present,
        "loaded_matches_current": live_loaded,
        "reason": reason,
        "process": _process_role(),
    }


def _process_role() -> str:
    """Which process observed this state: ``"server"`` or ``"worker"``.

    Read at call time from ``extension_companion``, which OWNS the answer, so a
    leaf that only projects liveness never has to import the loader it serves.
    The loader stamps the same key on its own reconcile receipts from the same
    owner, so the two can never disagree about who observed a state.
    """
    from ouroboros.extension_companion import is_server_process

    return "server" if is_server_process() else "worker"


def _deps_block_reason(drive_root: pathlib.Path, skill: LoadedSkill) -> str:
    """Return the dependency block reason, if live dispatch must refuse load."""
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash
        from ouroboros.marketplace.isolated_deps import read_deps_state
        from ouroboros.skill_dependencies import (
            auto_install_specs_for_skill,
            declared_dependency_names,
            payload_declared_install_specs,
        )

        auto_specs = auto_install_specs_for_skill(drive_root, skill)
        if not auto_specs:
            return ""
        # 6.2=A (ABI-1): the EFFECTIVE dependency set must match the names
        # declared by hash-covered payload carriers — an unhashed state-plane
        # record can never widen (or swap) the dependency surface silently.
        if declared_dependency_names(auto_specs) != declared_dependency_names(
            payload_declared_install_specs(skill)
        ):
            return "deps_declaration_desync"
        deps_state = read_deps_state(drive_root, skill.name, skill.skill_dir)
        status = str(deps_state.get("status") or "")
        if status != "installed":
            if status == "stale":
                return "deps_stale"
            return "deps_failed" if status == "failed" else "deps_missing"
        if deps_state.get("specs_hash") != install_specs_hash(auto_specs):
            return "deps_stale"
        return ""
    except Exception:
        log.debug("extension deps readiness probe failed", exc_info=True)
        return ""


def _apply_deps_block(state: Dict[str, Any], drive_root: pathlib.Path, skill: LoadedSkill) -> Dict[str, Any]:
    if state.get("desired_live"):
        deps_reason = _deps_block_reason(pathlib.Path(drive_root), skill)
        if deps_reason:
            state.update(desired_live=False, reason=deps_reason, load_error=deps_reason)
    return state


def _apply_durable_extension_health(
    state: Dict[str, Any], drive_root: pathlib.Path, skill: LoadedSkill
) -> Dict[str, Any]:
    from ouroboros.extension_health import apply_companion_failure_to_runtime_state

    return apply_companion_failure_to_runtime_state(state, drive_root, skill.name)


def runtime_state_for_skill_name(
    skill_name: str,
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
    skills: Optional[List[LoadedSkill]] = None,
) -> Dict[str, Any]:
    from ouroboros.config import get_skills_repo_path

    resolved_repo_path = get_skills_repo_path() if repo_path is None else repo_path
    peers = list(skills) if skills is not None else discover_skills(
        drive_root, repo_path=resolved_repo_path
    )
    safe_name = _sanitize_skill_name(skill_name)
    skill = next((item for item in peers if item.name == safe_name), None)
    if skill is None:
        with _lock:
            live_loaded = skill_name in _extensions
        state = {
            "skill": skill_name,
            "type": "extension",
            "runtime_mode": "",
            "enabled": False,
            "review_status": "missing",
            "review_stale": True,
            "load_error": "skill not found",
            "desired_live": False,
            "live_loaded": live_loaded,
            "loaded_present": live_loaded,
            "loaded_matches_current": False,
            "reason": "missing",
        }
    else:
        state = _apply_durable_extension_health(
            _apply_deps_block(
                _extension_runtime_state(
                    skill,
                    drive_root=pathlib.Path(drive_root),
                    skills=peers,
                    repo_path=resolved_repo_path,
                ),
                pathlib.Path(drive_root),
                skill,
            ),
            pathlib.Path(drive_root),
            skill,
        )
    # Stamped on BOTH branches: a missing skill is still an observation, and the
    # receipt must say which process made it.
    state["process"] = _process_role()
    return state


def runtime_state_for_loaded_skill(
    skill: "LoadedSkill",
    drive_root: pathlib.Path | None = None,
    *,
    skills: Optional[List[LoadedSkill]] = None,
) -> Dict[str, Any]:
    """Runtime state for an already-discovered skill; avoids repeated FS walks."""
    state = _extension_runtime_state(
        skill,
        drive_root=pathlib.Path(drive_root) if drive_root is not None else None,
        skills=skills,
    )
    if drive_root is None:
        return state
    root = pathlib.Path(drive_root)
    return _apply_durable_extension_health(_apply_deps_block(state, root, skill), root, skill)


def is_extension_live(
    skill_name: str,
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
) -> bool:
    state = runtime_state_for_skill_name(skill_name, drive_root, repo_path=repo_path)
    return bool(state.get("desired_live")) and bool(state.get("live_loaded"))


def _revert_enabled_after_load_error(
    revert: bool, drive_root: pathlib.Path, skill_name: str, state: Dict[str, Any]
) -> None:
    """Atomic enable: revert enabled.json to False when an enable-time load fails.

    Shared by every enable path (UI toggle, agent toggle_skill, post-review
    auto-enable) so a skill is never left enabled-but-broken regardless of who
    enabled it.
    """
    if not revert:
        return
    try:
        from ouroboros.skill_loader import save_enabled

        save_enabled(pathlib.Path(drive_root), skill_name, False, actor="load_error_revert")
        state["reverted_enabled"] = True
    except Exception:
        log.debug("Failed to revert enabled for %s after load error", skill_name, exc_info=True)
