from __future__ import annotations

import pathlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    next_actions: List[Dict[str, str]] = field(default_factory=list)


_SKILL_PAYLOAD_EDIT_TOOLS = frozenset({
    "write_file", "edit_text", "run_command", "run_script", "delegate_start",
})
_SKILL_PAYLOAD_SELECTOR_KEYS = {
    "run_command": "cwd", "run_script": "cwd", "delegate_start": "root",
}
_ROOT_TASK_PROJECTION_MAX_RECORDS = 512


def _skill_tool_identity_mapping() -> Dict[str, str]:
    """Live skill-tool name -> identity argument, derived from their schemas."""
    from ouroboros.tools.skill_exec import _EXEC_SCHEMA, _REVIEW_SCHEMA, _TOGGLE_SCHEMA, _OWNER_ACTION_SCHEMA
    from ouroboros.tools.skill_preflight import _PREFLIGHT_SCHEMA
    from ouroboros.tools.skill_publish import _PUBLISH_SCHEMA

    schemas = (_REVIEW_SCHEMA, _PREFLIGHT_SCHEMA, _EXEC_SCHEMA, _TOGGLE_SCHEMA, _PUBLISH_SCHEMA, _OWNER_ACTION_SCHEMA)
    return {
        str(schema["name"]): str(schema["parameters"]["required"][0])
        for schema in schemas
    }


def skill_names_touched_by_trace(llm_trace: Dict[str, Any]) -> List[str]:
    """Names of the skills a task's tool trace touched.

    Payload edits name the skill through ``bucket``/``skill_name`` or the path;
    the skill lifecycle tools name it directly, which is the only carrier for a
    delegated payload the root never wrote with ``write_file``/``edit_text``.
    """
    names: List[str] = []
    identity_keys = _skill_tool_identity_mapping()
    for call in llm_trace.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "")
        if tool not in _SKILL_PAYLOAD_EDIT_TOOLS and tool not in identity_keys:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if tool in identity_keys:
            named = str(args.get(identity_keys[tool]) or "").strip()
            if named and named not in names:
                names.append(named)
            continue
        selector_key = _SKILL_PAYLOAD_SELECTOR_KEYS.get(tool)
        if selector_key:
            selector = str(args.get(selector_key) or "").replace("\\", "/").rstrip("/")
            if not (
                selector == "skill_payload"
                or (selector_key == "cwd" and selector.startswith("skill_payload/"))
            ):
                continue
        bucket = str(args.get("bucket") or "").strip().lower()
        skill_name = str(args.get("skill_name") or "").strip()
        if bucket in {"external", "clawhub", "ouroboroshub", "user_repo"} and skill_name:
            if skill_name not in names:
                names.append(skill_name)
            continue
        candidates = [str(args.get("path") or "")]
        for raw in candidates:
            norm = raw.replace("\\", "/").strip().lstrip("/")
            if norm.startswith("data/"):
                norm = norm[len("data/"):]
            parts = pathlib.PurePosixPath(norm).parts
            if len(parts) >= 3 and parts[0] == "skills" and parts[1] in {"external", "clawhub", "ouroboroshub", "native"}:
                name = parts[2]
                if name and name not in names:
                    names.append(name)
    return names


def acceptance_skill_lifecycle(
    drive_root: Any, llm_trace: Dict[str, Any], root_task_id: str = "",
    task_started_at: str = "",
    history_coverage: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Per-skill lifecycle facts of the skills a task touched — VISIBILITY ONLY.

    The task's completion observations
    (``task_finalization.build_completion_observations``) read them. Names come
    from the task's own trace plus any skill whose review history records this
    ``root_task_id`` (which is how a child-authored payload shows up at all). A
    UI or manual review carries no root task id and is simply not joined.
    Nothing here is a gate.
    """
    root = pathlib.Path(drive_root) if drive_root else None
    if root is None:
        return []
    names = list(skill_names_touched_by_trace(llm_trace or {}))
    history = _skill_names_from_review_history(
        root, str(root_task_id or ""), task_started_at=task_started_at,
    )
    if history_coverage is not None:
        history_coverage.update(history["coverage"])
    for name in history["names"]:
        if name not in names:
            names.append(name)
    if not names:
        return []
    from ouroboros.skill_loader import discover_skills, find_skill

    peers = discover_skills(root)
    rows: List[Dict[str, Any]] = []
    for name in names:
        skill = find_skill(root, name)
        if skill is None:
            rows.append({"name": name, "present": False})
            continue
        readiness = skill_readiness_for_execution(root, skill, skills=peers)
        runtime = None
        if skill.manifest.is_extension():
            try:
                from ouroboros.extension_loader import runtime_state_for_loaded_skill

                live = runtime_state_for_loaded_skill(skill, root, skills=peers)
                runtime = {key: live.get(key) for key in ("desired_live", "live_loaded", "reason", "load_error", "process")}
            except Exception as exc:
                runtime = {"live_loaded": None, "reason": "unavailable", "load_error": str(exc)}
        rows.append({
            "name": skill.name,
            "source": str(getattr(skill, "source", "") or ""),
            "content_hash": skill.content_hash,
            "review_status": str(getattr(skill.review, "status", "") or ""),
            "review_stale": bool(skill.review.is_stale_for(skill.content_hash)),
            "review_gate": skill.review.gate_for(skill.content_hash),
            "enabled": bool(getattr(skill, "enabled", False)),
            "ready": bool(readiness.ready),
            "blockers": list(readiness.blockers),
            "manual_dependencies": list(readiness.manual_dependencies),
            "grants": dict(readiness.grant_status),
            "next_actions": list(readiness.next_actions),
            "extension_runtime": runtime,
        })
    return rows


def _skill_names_from_review_history(
    drive_root: pathlib.Path, root_task_id: str, *, task_started_at: str = "",
) -> Dict[str, Any]:
    """The skills this root task reviewed, named by the review ledger's bounded hot index (one row per
    ``surface="skill"`` record, its skill in ``subject.skill``), never by full records or histories."""
    from ouroboros import review_ledger as ledger

    source_ref = {"kind": "review_ledger_index", "path": f"state/{ledger.LEDGER_SUBDIR}/{ledger.INDEX_NAME}",
                  "reader": "read_file"}
    gaps: set[str] = set()
    names: List[str] = []
    rows: List[Dict[str, Any]] = []
    cutoff = str(task_started_at or "").replace("Z", "+00:00")
    if root_task_id:
        try:
            rows = ledger.recent_records(drive_root, root_task_id, limit=_ROOT_TASK_PROJECTION_MAX_RECORDS,
                                         hot_only=True, surface="skill")
        except OSError:
            gaps.add("projection_unreadable")
        if ledger.archived_segments_exist(drive_root):
            # Older records live in archived segments the hot read never opens: a cut only when this task
            # could have rows there — it started before the index last rotated, or no start is known; a task
            # that started after the newest rotation has every row of its own in the hot index.
            rotated = ledger.newest_rotation_ts(drive_root)
            if not cutoff or not rotated or cutoff <= rotated:
                gaps.add("tail_bytes_truncated")
    for row in rows:
        row_ts = str(row.get("ts") or "").replace("Z", "+00:00")
        if cutoff and row_ts and row_ts < cutoff:
            continue
        name = str((row.get("subject") or {}).get("skill") or "").strip()
        if name and name not in names:
            names.append(name)
    coverage = {"rows_scanned": len(rows), "gap_rows_scanned": 0, "truncated": "tail_bytes_truncated" in gaps,
                "complete": not gaps, "gap_reasons": sorted(gaps), "source_ref": source_ref}
    return {"names": names, "coverage": coverage}


# Unready packages are installed by the enable owner, never by a paid review: whatever admits the
# bytes admits their declared dependencies (skill_dependencies.ensure_skill_dependencies).
_DEPENDENCY_ACTION = {"phase": "dependencies", "tool": "toggle_skill", "reason": (
    "toggle_skill enabled=true installs or refreshes the declared dependencies when the review gate admits "
    "the current bytes; it repairs an enabled skill too. Enable only where owner intent covers it, "
    "never over an owner's OFF.")}


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
    next_actions: List[Dict[str, str]] = []

    if getattr(skill, "load_error", ""):
        msg = f"load_error={skill.load_error!r}"
        blockers.append(msg)
        agent_fixable.append(msg)
        next_actions.append({"phase": "payload", "tool": "skill_preflight", "reason": msg})

    stale = skill.review.is_stale_for(skill.content_hash)
    gate = skill.review.gate_for(skill.content_hash)
    if stale and not gate["executable_review"]:
        blockers.append("review_stale")
        agent_fixable.append("review_stale")
        next_actions.append({"phase": "review", "tool": "skill_review", "reason": "review_stale"})
    elif not gate.get("executable_review"):
        reason = str(gate.get("blocking_reason") or "review_not_executable")
        msg = f"review_not_executable:{reason}"
        blockers.append(msg)
        agent_fixable.append(msg)
        next_actions.append({"phase": "review", "tool": "skill_review", "reason": msg})

    if require_enabled and not getattr(skill, "enabled", False):
        blockers.append("skill_disabled")
        owner_action.append("skill_disabled")
        next_actions.append({"phase": "enablement", "tool": "toggle_skill", "reason": "Use the existing owner request when it covers enabling; Repair and run needs no new approval. Preserve later owner disablement."})

    from ouroboros.skill_loader import discover_skills, skill_conflict_status

    peers = skills if skills is not None else discover_skills(pathlib.Path(drive_root))
    conflict = skill_conflict_status(skill, peers) or {}
    if conflict:
        names = list(conflict.get("skills") or [])
        suffix = f":{','.join(names)}" if names else ""
        msg = f"skill_conflict{suffix}"
        blockers.append(msg)
        owner_action.append(msg)
        next_actions.append({"phase": "conflict", "tool": "toggle_skill", "reason": "Resolve conflicting enablement according to owner intent."})

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
            next_actions.append({"phase": "grants", "tool": "skill_owner_action", "reason": "Grant only manifest items covered by expressed owner intent; repeating review does not supply owner permission."})

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
                next_actions.append(dict(_DEPENDENCY_ACTION))
            elif deps_state.get("specs_hash") != install_specs_hash(auto_specs):
                blockers.append("deps_stale")
                agent_fixable.append("deps_stale")
                next_actions.append(dict(_DEPENDENCY_ACTION))
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
        next_actions=next_actions,
    )
