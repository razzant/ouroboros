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


_SKILL_PAYLOAD_EDIT_TOOLS = frozenset({
    "write_file", "edit_text", "run_command", "run_script", "delegate_start",
})
_SKILL_PAYLOAD_SELECTOR_KEYS = {
    "run_command": "cwd", "run_script": "cwd", "delegate_start": "root",
}
_ROOT_TASK_PROJECTION_MAX_BYTES = 1024 * 1024
_ROOT_TASK_PROJECTION_MAX_RECORDS = 512


def _skill_tool_identity_mapping() -> Dict[str, str]:
    """Live skill-tool name -> identity argument, derived from their schemas."""
    from ouroboros.tools.skill_exec import _EXEC_SCHEMA, _REVIEW_SCHEMA, _TOGGLE_SCHEMA
    from ouroboros.tools.skill_preflight import _PREFLIGHT_SCHEMA
    from ouroboros.tools.skill_publish import _PUBLISH_SCHEMA

    schemas = (_REVIEW_SCHEMA, _PREFLIGHT_SCHEMA, _EXEC_SCHEMA, _TOGGLE_SCHEMA, _PUBLISH_SCHEMA)
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
    """Per-skill lifecycle facts for the acceptance packet — VISIBILITY ONLY.

    Names come from the task's own trace plus any skill whose review history
    records this ``root_task_id`` (which is how a child-authored payload shows
    up at all). A UI or manual review carries no root task id and is simply not
    joined. Acceptance judges quality, never the execution route, so nothing
    here is a gate.
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
        rows.append({
            "name": skill.name,
            "source": str(getattr(skill, "source", "") or ""),
            "review_status": str(getattr(skill.review, "status", "") or ""),
            "review_stale": bool(skill.review.is_stale_for(skill.content_hash)),
            "enabled": bool(getattr(skill, "enabled", False)),
            "ready": bool(readiness.ready),
            "blockers": list(readiness.blockers),
            "manual_dependencies": list(readiness.manual_dependencies),
        })
    return rows


def _skill_names_from_review_history(
    drive_root: pathlib.Path, root_task_id: str, *, task_started_at: str = "",
) -> Dict[str, Any]:
    """Skills named by the compact root-task projection, never full histories."""
    from ouroboros.skill_review_history import (
        ROOT_TASK_PROJECTION_GAPS_RELATIVE_PATH, ROOT_TASK_PROJECTION_RELATIVE_PATH,
        root_task_projection_gaps_path, root_task_projection_path,
    )
    from ouroboros.utils import iter_jsonl_objects

    path = root_task_projection_path(drive_root)
    gap_path = root_task_projection_gaps_path(drive_root)
    source_ref = {
        "kind": "canonical_jsonl", "path": ROOT_TASK_PROJECTION_RELATIVE_PATH,
        "reader": "read_file",
    }
    gaps: set[str] = set()
    try:
        byte_truncated = path.stat().st_size > _ROOT_TASK_PROJECTION_MAX_BYTES
    except FileNotFoundError:
        byte_truncated = False
    except OSError:
        byte_truncated = False
        gaps.add("projection_unreadable")
    names: List[str] = []
    try:
        rows = list(iter_jsonl_objects(
            path,
            max_entries=_ROOT_TASK_PROJECTION_MAX_RECORDS,
            tail_bytes=_ROOT_TASK_PROJECTION_MAX_BYTES,
            gap_reasons=gaps,
        ))
    except OSError:
        rows = []
        gaps.add("projection_unreadable")
    if byte_truncated:
        gaps.add("tail_bytes_truncated")
    try:
        gap_byte_truncated = gap_path.stat().st_size > _ROOT_TASK_PROJECTION_MAX_BYTES
    except FileNotFoundError:
        gap_byte_truncated = False
    except OSError:
        gap_byte_truncated = False
        gaps.add("projection_gap_log_unreadable")
    try:
        gap_rows = list(iter_jsonl_objects(
            gap_path,
            max_entries=_ROOT_TASK_PROJECTION_MAX_RECORDS,
            tail_bytes=_ROOT_TASK_PROJECTION_MAX_BYTES,
            gap_reasons=gaps,
        ))
    except OSError:
        gap_rows = []
        gaps.add("projection_gap_log_unreadable")
    if gap_byte_truncated:
        gaps.add("projection_gap_tail_bytes_truncated")
    cutoff = str(task_started_at or "").replace("Z", "+00:00")
    if root_task_id:
        for row in reversed(rows):
            row_ts = str(row.get("ts") or "").replace("Z", "+00:00")
            if cutoff and row_ts and row_ts < cutoff:
                break
            if not isinstance(row, dict) or str(row.get("root_task_id") or "") != root_task_id:
                continue
            name = str(row.get("skill") or "").strip()
            if name and name not in names:
                names.append(name)
        for row in reversed(gap_rows):
            row_ts = str(row.get("ts") or "").replace("Z", "+00:00")
            if cutoff and row_ts and row_ts < cutoff:
                break
            if isinstance(row, dict) and str(row.get("root_task_id") or "") == root_task_id:
                gaps.add(str(row.get("reason") or "root_task_projection_gap"))
    coverage = {
        "rows_scanned": len(rows), "gap_rows_scanned": len(gap_rows),
        "truncated": bool(byte_truncated or gap_byte_truncated or "max_entries_truncated" in gaps),
        "complete": not gaps, "gap_reasons": sorted(gaps), "source_ref": source_ref,
        "gap_source_ref": {
            "kind": "canonical_jsonl", "path": ROOT_TASK_PROJECTION_GAPS_RELATIVE_PATH,
            "reader": "read_file",
        },
    }
    return {"names": names, "coverage": coverage}


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
