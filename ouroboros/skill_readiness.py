from __future__ import annotations

import datetime as _dt
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


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
    # Pending custom keys are not passed but do not block by themselves (``skill_grant_follow.grant_notes``).
    notes: List[str] = field(default_factory=list)
    next_actions: List[Dict[str, str]] = field(default_factory=list)


_SKILL_PAYLOAD_EDIT_TOOLS = frozenset({
    "write_file", "edit_text", "run_command", "run_script", "delegate_start",
})
_SKILL_FOLLOWUP_TOOLS = _SKILL_PAYLOAD_EDIT_TOOLS | {"skill_review", "toggle_skill"}
_SKILL_PAYLOAD_SELECTOR_KEYS = {
    "run_command": "cwd", "run_script": "cwd", "delegate_start": "root",
}
_ROOT_TASK_PROJECTION_MAX_RECORDS = 512


def _grant_stamp(iso: str) -> str:
    try:
        when = _dt.datetime.fromisoformat(iso).astimezone(_dt.timezone.utc)
    except (TypeError, ValueError):
        return ""
    return f" (grant last updated {when.strftime('%Y-%m-%d %H:%M')} UTC)"


def grant_gap_sentence(grants: Any) -> str:
    """The plain-words cause behind a missing grant, shared by every producer that names one (the readiness
    line, the execution refusal, the toggle refusal): the owner's grant exists for an earlier version of the
    skill's files (``grant_status_for_skill`` ``earlier_grant``). Empty when nothing beyond "not granted yet"
    is known. Never a hash: the mind and the owner read versions of files, not digests."""
    earlier = grants.get("earlier_grant") if isinstance(grants, Mapping) else None
    if not isinstance(earlier, Mapping):
        return ""
    items = [str(item) for item in list(earlier.get("keys") or []) + list(earlier.get("permissions") or [])]
    if not items:
        return ""
    names = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    verb = "was" if len(items) == 1 else "were"
    return (f"{names} {verb} granted to an earlier version of this skill's files{_grant_stamp(str(grants.get('updated_at') or ''))}; "
            "the files changed since, so that grant does not cover the current files. The saved Settings values are untouched.")


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


def note_enabled_skill_digests(ctx: Any, skills: Any) -> None:
    """Keep the first enabled digests and changes seen on later catalog reads in this ToolContext, not across workers."""
    if ctx is None or not isinstance(skills, list):
        return
    start = getattr(ctx, "_skill_digests_at_start", None)
    if start is None:
        ctx._skill_digests_at_start = {
            str(skill.name): str(skill.content_hash) for skill in skills or []
            if getattr(skill, "enabled", False) and getattr(skill, "content_hash", "")
            and not getattr(skill, "identity_collision", False)}
    else:
        ctx._skill_changes_observed = getattr(ctx, "_skill_changes_observed", set()) | {
            skill.name for skill in skills if not skill.identity_collision and skill.content_hash
            and skill.name in start and skill.content_hash != start[skill.name]}


def observe_named_skills(drive_root: pathlib.Path, names: List[str]) -> tuple[Dict[str, Any], List[Any]]:
    """One location/peer view for named subjects; only those subjects need payload identity."""
    if not names:
        return {}, []
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import _skill_location_inventory, discover_skill_identity
    from ouroboros.skill_peer_inventory import discover_skill_peers

    repo_path = get_skills_repo_path() or None
    locations = _skill_location_inventory(drive_root, repo_path=repo_path)
    subjects: Dict[str, Any] = {}
    for name in names:
        try:
            if selected := discover_skill_identity(drive_root, name, repo_path=repo_path, locations=locations):
                subjects[name] = selected[0]
        except Exception:
            log.debug("Skill observation unavailable for %s", name, exc_info=True)
    peers = discover_skill_peers(drive_root, repo_path=repo_path, locations=locations) if subjects else ()
    return subjects, list(peers)


def _diagnostic_preview(text: Any, source: str, limit: int = 300) -> str:
    from ouroboros.utils import truncate_within_limit

    text = str(text or "").strip()
    return text if len(text) <= limit else f"{truncate_within_limit(text, limit)}; more at {source}"


def _runtime_diagnostic(drive_root: pathlib.Path, skill: Any, state: Dict[str, Any], reader: str) -> str:
    """An in-process load attempt matches the current files; retained server health has no payload digest."""
    from ouroboros.extension_health import (
        BROKEN, COMPANION_FAILED, COMPANION_RESTART_EXHAUSTED, read_extension_health,
    )

    if state.get("reason") == "load_error":
        detail = _diagnostic_preview(state.get("load_error"), reader)
        return "a load attempt for the current extension failed in this process" + (f": {detail}" if detail else "")
    if state.get("foreign_publication"):
        return "this installation's server health could not be verified"

    health_reader = f"query_data(op='json', kind='skill_state', ids=[{skill.name!r}], state_file='health')"
    if state.get("process") == "server":
        if state.get("companion_failed"):
            _, _, names = str(state.get("reason") or "").partition(":")
            component = f" ({names})" if names else ""
            detail = _diagnostic_preview(state.get("load_error") or state.get("reason"), health_reader)
            return f"the server reports a companion failure{component} (payload revision not established): {detail}"
        return ""

    server = ((read_extension_health(drive_root, skill.name) or {}).get("observations") or {}).get("server")
    if not isinstance(server, dict) or server.get("status") != BROKEN:
        return ""
    reason = str(server.get("reason") or "")
    prefix, _, component = reason.partition(":")
    if prefix in {COMPANION_FAILED, COMPANION_RESTART_EXHAUSTED}:
        subject = f"companion failure ({component})" if component else "companion failure"
    else:
        subject = "extension load failure" if reason == "load_error" else "extension health failure"
    detail = _diagnostic_preview(server.get("load_error") or reason, health_reader)
    return f"last recorded server {subject} (payload revision not recorded)" + (f": {detail}" if detail else "")


def _skill_status_fact(drive_root: pathlib.Path, skill: Any, peers: List[Any], *, warned: bool = False) -> str:
    """Configuration, then process evidence: never infer runtime success from a grant or from an old health record."""
    from ouroboros.skill_catalogue import skill_status
    from ouroboros.skill_grant_follow import files_accepted

    readiness = skill_readiness_for_execution(drive_root, skill, skills=peers)
    if warned and getattr(skill, "is_self_authored", False) and not readiness.ready:
        return ""
    state, runtime_note = {}, ""
    reader = f"list_skills(name={skill.name!r})"
    if readiness.ready and skill.manifest.is_extension():
        from ouroboros.extension_liveness import runtime_state_for_loaded_skill

        state = runtime_state_for_loaded_skill(skill, drive_root, skills=peers)
        if state.get("desired_live"):
            runtime_note = _runtime_diagnostic(drive_root, skill, state, reader)
    status = skill_status({"type": skill.manifest.type,
                           "readiness": {"blockers": readiness.blockers, "notes": readiness.notes},
                           "entry_gap": skill.entry_gap if readiness.ready and skill.manifest.is_script() else "",
                           "declares_mcp": bool(skill.manifest.mcp_servers),
                           "desired_live": state.get("desired_live"), "live_reason": state.get("reason"),
                           "load_error": state.get("load_error")})
    facts = []
    if status["reasons"]:
        facts.append("not ready: " + _diagnostic_preview("; ".join(status["reasons"]), reader))
        if readiness.next_actions:
            facts.append("next actions: " + _diagnostic_preview(readiness.next_actions, reader, limit=480))
    elif not files_accepted(skill.review, skill.content_hash):
        facts.append("no recorded acceptance or clean/warnings review for the current files")
    if status["notes"]:
        facts.append(_diagnostic_preview("; ".join(status["notes"]), reader))
    if runtime_note:
        facts.append(runtime_note)
    return "; ".join(facts)


def changed_skill_facts(drive_root: pathlib.Path, llm_trace: Dict[str, Any], ctx: Any) -> List[str]:
    """Follow up only named subjects or differences seen on an existing catalog read; no finish-wide search.

    Tool arguments tell us where to look, never whether files changed or who changed them. Off stays quiet.
    Untargeted edits after the last existing observation and edits before this ToolContext's first view are unknown.
    """
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    start = getattr(ctx, "_skill_digests_at_start", None) or {}
    trace = llm_trace or {}
    touched = skill_names_touched_by_trace(trace)
    selected = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    names = list(dict.fromkeys([
        *skill_names_for_followup(trace), *(name for name in touched if name in start),
        *sorted(getattr(ctx, "_skill_changes_observed", ())),
        *([selected.skill_name] if selected and selected.has_selected_skill else []),
    ]))
    facts: List[str] = []
    try:
        subjects, peers = observe_named_skills(drive_root, names)
    except Exception:
        log.debug("Skill follow-up view unavailable", exc_info=True)
        return []
    for name, skill in subjects.items():
        if not skill.enabled or skill.identity_collision:
            continue
        before = start.get(skill.name)
        if before and skill.content_hash:
            if before == skill.content_hash:
                continue
            lead = f"files of skill {skill.name} changed since this work's first observation (author not established)"
        else:
            lead = f"current state of skill {skill.name} (file changes not verified)"
        try:
            warned = name in touched and getattr(ctx, "_skill_finalization_injected", False)
            if fact := _skill_status_fact(drive_root, skill, peers, warned=warned):
                facts.append(f"{lead}; {fact}")
        except Exception:
            log.debug("Skill follow-up fact unavailable for %s", skill.name, exc_info=True)
    return facts


def changed_skills_notice(drive_root: pathlib.Path, llm_trace: Dict[str, Any], ctx: Any) -> str:
    """A neutral end-of-work line, compared once per ToolContext whether or not it finds a fact."""
    if ctx is None or getattr(ctx, "_skill_change_compared", False):
        return ""
    ctx._skill_change_compared = True
    facts = changed_skill_facts(pathlib.Path(drive_root), llm_trace, ctx)
    return "Skill status observations: " + "; ".join(facts) if facts else ""


def skill_names_for_followup(llm_trace: Dict[str, Any]) -> List[str]:
    """Subjects named by an attempted edit, review or toggle, not evidence of an effect."""
    calls = [call for call in llm_trace.get("tool_calls") or []
             if isinstance(call, dict) and str(call.get("tool") or "") in _SKILL_FOLLOWUP_TOOLS]
    return skill_names_touched_by_trace({"tool_calls": calls})


def skill_names_touched_by_trace(llm_trace: Dict[str, Any]) -> List[str]:
    """Names of the skills a task's tool trace touched.

    Payload edits name the skill through ``bucket``/``skill_name`` or the path;
    the skill lifecycle tools name it directly, which is the only carrier for a
    delegated payload the root never wrote with ``write_file``/``edit_text``.
    """
    calls = llm_trace.get("tool_calls") or []
    if not calls:
        return []
    names: List[str] = []
    identity_keys = _skill_tool_identity_mapping()
    for call in calls:
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
    subjects, peers = observe_named_skills(root, names)
    rows: List[Dict[str, Any]] = []
    for name in names:
        skill = subjects.get(name)
        if skill is None:
            rows.append({"name": name, "present": False})
            continue
        readiness = skill_readiness_for_execution(root, skill, skills=peers)
        runtime = None
        if skill.manifest.is_extension():
            try:
                from ouroboros.extension_loader import runtime_state_for_loaded_skill

                live = runtime_state_for_loaded_skill(skill, root, skills=peers)
                runtime = {key: live.get(key) for key in ("desired_live", "live_loaded", "reason", "load_error", "process",
                                                          "grant_note")}
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
    notes: List[str] = []
    if require_grants:
        from ouroboros.skill_grant_follow import blocking_lists, grant_notes
        from ouroboros.skill_loader import grant_status_for_skill

        grants = grant_status_for_skill(pathlib.Path(drive_root), skill)
        notes = grant_notes(grants)
        if grants.get("blocked"):
            missing_keys, missing_permissions = blocking_lists(grants)
            cause = grant_gap_sentence(grants)
            msg = f"missing_grants:keys={missing_keys},permissions={missing_permissions}" + (f"; {cause}" if cause else "")
            blockers.append(msg)
            owner_action.append(msg)
            reason = ("Use skill_owner_action grant for items the owner actually approved. With automatic grants on, "
                      "an executable review, qualified author finish, owner Skip, or accepted skill_install/skill_update "
                      "can grant declared items for these files. A recorded Repair edit admits files but grants nothing.")
            if cause:
                reason += " The grant for earlier files does not cover the current files. Settings values stay saved."
            next_actions.append({"phase": "grants", "tool": "skill_owner_action", "reason": reason})

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
        notes=notes,
        next_actions=next_actions,
    )
