"""Bounded facts one owner turn is allowed to address.

Projections only: which root tasks a chat can steer, what a project's last
result says about where its work lives, what the Main lane can see, and how a
chat maps to a project. Nothing here delivers a message or picks a target —
that judgment belongs to the decision turn (BIBLE P5).
"""

import pathlib
from typing import Any, Dict, Optional

from ouroboros.server_process import log


def _task_belongs_to_chat(ctx: Any, task_id: str, task_obj: Dict[str, Any], chat_id: int) -> bool:
    try:
        if int(task_obj.get("chat_id") or 0) == int(chat_id or 0):
            return True
    except (TypeError, ValueError):
        pass
    try:
        from ouroboros.projects_registry import project_chat_for_task

        return int(project_chat_for_task(ctx.DRIVE_ROOT, task_id) or 0) == int(chat_id or 0)
    except Exception:
        return False


def _active_direct_root(ctx: Any) -> Dict[str, Any]:
    """Snapshot the one in-process direct root without creating queue state."""
    try:
        agent = ctx.get_chat_agent()
        lock = getattr(agent, "_owner_message_admission_lock", None)
        if lock is None:
            return {}
        with lock:
            task_id = str(getattr(agent, "_current_task_id", "") or "").strip()
            if (
                not getattr(agent, "_busy", False)
                or not getattr(agent, "_accepting_owner_messages", False)
                or not task_id
            ):
                return {}
            metadata = getattr(agent, "_current_task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            return {
                "task_id": task_id,
                "status": "running",
                "title": _clip_marked(metadata.get("title"), 120),
                "objective": _clip_marked(getattr(agent, "_current_task_text", ""), 600),
                "project_id": str(metadata.get("project_id") or ""),
                "chat_id": int(getattr(agent, "_current_chat_id", 0) or 0),
                "started_at": float(getattr(agent, "_task_started_ts", 0.0) or 0.0),
                "steerable": True,
                "direct_chat": True,
            }
    except Exception:
        return {}


def _addressable_root_tasks(ctx: Any, chat_id: Optional[int] = None) -> list:
    """Compact RUNNING+PENDING owner-root manifest, without choosing a target."""
    out: list = []
    seen: set[str] = set()

    def _add(task_id: Any, task_obj: Any, status: str, started_at: Any = None) -> None:
        tid = str(task_id or "").strip()
        if not tid or tid in seen or not isinstance(task_obj, dict):
            return
        if task_obj.get("_is_direct_chat") or str(task_obj.get("delegation_role") or "") == "subagent":
            return
        if chat_id is not None and not _task_belongs_to_chat(ctx, tid, task_obj, int(chat_id or 0)):
            return
        objective = str(
            task_obj.get("objective") or task_obj.get("description") or task_obj.get("text") or ""
        ).strip()
        out.append({
            "task_id": tid,
            "status": status,
            "title": _clip_marked(task_obj.get("title"), 120),
            "objective": _clip_marked(objective, 600),
            "project_id": str(task_obj.get("project_id") or ""),
            "started_at": started_at,
            "steerable": True,
        })
        seen.add(tid)

    for tid, running in list(getattr(ctx, "RUNNING", {}).items()):
        if not isinstance(running, dict):
            continue
        task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
        _add(tid, task_obj, "running", running.get("started_at"))
    for pending in list(getattr(ctx, "PENDING", []) or []):
        if isinstance(pending, dict):
            _add(pending.get("id"), pending, "pending", pending.get("queued_at"))
    direct = _active_direct_root(ctx)
    if direct and str(direct.get("task_id") or "") not in seen:
        if chat_id is None or int(direct.get("chat_id") or 0) == int(chat_id or 0):
            out.append(direct)
    return out


def _clip_marked(value: str, limit: int) -> str:
    """Clip a routing/recognition string but NEVER silently: an explicit omission
    marker keeps a decision-context field honest (no silent ``[:N]`` truncation of a
    cognitive/routing artifact — DEVELOPMENT.md). The marker + the full task_id keep
    enough signal for the agent to disambiguate the steer target."""
    s = str(value or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[+{len(s) - limit} chars omitted]"


def _chat_running_tasks(ctx: Any, chat_id: int) -> list:
    """Structural snapshot of the owner's RUNNING root tasks in THIS chat (id +
    objective + recency). The decision turn reads this from runtime context to
    pick a steer_task target by its own judgment — code only exposes the state,
    it never auto-chooses (BIBLE P5). Direct in-process turns and subagents are
    not pooled RUNNING tasks and are excluded."""
    return [row for row in _addressable_root_tasks(ctx, chat_id) if row.get("status") == "running"]


def _task_result_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded typed projection of one task result for a routing/promote turn:
    identity, outcome, and WHERE THE WORK LIVES (workspace facts + artifact refs).
    Never raw result text — a router turn that reconstructs prior work from chat
    memory instead of these facts invents false premises (the saga's "continue"
    promotion rebuilt a finished game from scratch)."""
    bundle = row.get("artifact_bundle") if isinstance(row.get("artifact_bundle"), dict) else {}
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), list) else []
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    out = {
        "task_id": str(row.get("task_id") or row.get("id") or ""),
        "status": str(row.get("status") or ""),
        "title": _clip_marked(row.get("title"), 120),
        "objective": _clip_marked(row.get("objective") or row.get("description"), 300),
        "project_id": str(row.get("project_id") or ""),
        "reason_code": str(row.get("reason_code") or ""),
        "workspace_root": str(row.get("workspace_root") or ""),
        "workspace_mode": str(row.get("workspace_mode") or ""),
        "artifact_status": str(row.get("artifact_status") or ""),
        "artifact_refs": [
            str(item.get("path") or item.get("name") or "")
            for item in artifacts[:8] if isinstance(item, dict)
        ],
    }
    if git:
        out["workspace_git_at_start"] = {
            "head": str(git.get("head") or ""),
            "branch": str(git.get("branch") or ""),
            "dirty": bool(git.get("dirty")),
        }
    return out


def _latest_project_task_result(ctx: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Newest task result bound to ``project_id`` WITHOUT replaying the whole
    store (DEVELOPMENT "Projection over replay"). The registry row's durable
    ``last_task_result_id`` pointer (stamped at project-task finalization) is
    read FIRST — one direct file fetch, immune to how many newer foreign
    results exist. Only when the pointer is absent or stale (missing/
    unparseable/foreign file) does the fallback run: the bounded newest-64
    mtime scan, then — for pre-pointer projects only — a disclosed full scan
    of the store (the lazy self-heal for rows finalized before the pointer
    existed; with zero matching results nothing is written back, so it repeats
    per lookup until a matching result exists). Only the ABSENT-pointer case
    writes the pointer back: a non-empty pointer that failed to resolve is
    usually a split-drive result in flight (finalization stamps the pointer
    before the canonical copy-back lands), so overwriting it from the scan
    would permanently regress it to an older result — serve the scan hit and
    let the pointer resolve itself. The steady state needs no
    ouroboros/context_budget.py threshold enrollment (that table guards
    recurring full-store replays)."""
    from ouroboros.projects_registry import get_project, update_project
    from ouroboros.task_results import load_task_result, task_results_dir
    from ouroboros.utils import read_json_dict

    try:
        pointer = str((get_project(ctx.DRIVE_ROOT, project_id) or {}).get(
            "last_task_result_id") or "").strip()
    except Exception:
        pointer = ""
    if pointer:
        pointed = load_task_result(ctx.DRIVE_ROOT, pointer)
        if isinstance(pointed, dict) and str(pointed.get("project_id") or "") == project_id:
            return pointed
        log.debug(
            "project last-task-result pointer for %r is stale (%s); "
            "falling back to the bounded scan", project_id, pointer,
        )

    paths = list(task_results_dir(ctx.DRIVE_ROOT, create=False).glob("*.json"))
    try:
        paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        paths.sort(key=lambda path: path.name, reverse=True)
    row = None
    for path in paths[:64]:
        candidate = read_json_dict(path)
        if candidate is not None and str(candidate.get("project_id") or "") == project_id:
            row = candidate
            break
    if row is None and len(paths) > 64:
        log.info(
            "project last-task-result: %r missed the bounded scan; running the "
            "full-store self-heal scan (%d files)", project_id, len(paths),
        )
        for path in paths[64:]:
            candidate = read_json_dict(path)
            if candidate is not None and str(candidate.get("project_id") or "") == project_id:
                row = candidate
                break
    if row is not None and not pointer:
        try:
            update_project(ctx.DRIVE_ROOT, project_id, last_task_result_id=str(
                row.get("task_id") or row.get("id") or ""))
        except Exception:
            log.debug("project last-task-result pointer write-back failed", exc_info=True)
    return row


def _main_routing_manifest(ctx: Any) -> Dict[str, Any]:
    """Bounded canonical facts for one Main-chat LLM routing decision."""
    from ouroboros.projects_registry import list_projects
    from ouroboros.task_results import list_task_results
    from ouroboros.utils import iter_jsonl_objects

    projects = [{
        "project_id": str(row.get("id") or ""),
        "name": _clip_marked(row.get("name"), 120),
        "chat_id": int(row.get("chat_id") or 0),
        "lifecycle": str(row.get("lifecycle") or "active"),
        # Registry-canonical working folder: the router turn's ground truth for
        # where a project's work lives (Q8-A).
        "working_dir": str(row.get("working_dir") or ""),
    } for row in list_projects(ctx.DRIVE_ROOT)]
    roots = _addressable_root_tasks(ctx, None)

    all_results = list_task_results(ctx.DRIVE_ROOT)
    all_results.sort(key=lambda row: str(row.get("ts") or row.get("updated_at") or ""), reverse=True)
    finals = [_task_result_ground_truth(row) for row in all_results[:16]]

    dialogue_rows: list = []
    chat_paths = sorted(
        (pathlib.Path(ctx.DRIVE_ROOT) / "archive").glob("chat_*.jsonl"),
        key=lambda path: path.name,
    )[-2:] + [pathlib.Path(ctx.DRIVE_ROOT) / "logs" / "chat.jsonl"]
    for path in chat_paths:
        for row in iter_jsonl_objects(path):
            text = str(row.get("text") or "").strip()
            if text:
                dialogue_rows.append({
                    "ts": str(row.get("ts") or ""),
                    "direction": str(row.get("direction") or ""),
                    "chat_id": int(row.get("chat_id") or 1),
                    "text": _clip_marked(text, 500),
                    "task_id": str(row.get("task_id") or ""),
                    "client_message_id": str(row.get("client_message_id") or ""),
                })
    dialogue = dialogue_rows[-20:]
    return {
        "projects": projects[:40],
        "root_tasks": roots[:40],
        "final_results": finals,
        "recent_canonical_dialogue": dialogue,
        "omissions": {
            "projects": max(0, len(projects) - 40),
            "root_tasks": max(0, len(roots) - 40),
            "final_results": max(0, len(all_results) - 16),
            "dialogue_rows": max(0, len(dialogue_rows) - 20),
        },
    }


def _decision_turn_metadata(ctx: Any, chat_id: int, client_message_id: str, task_metadata: Any) -> Any:
    """Enrich a chat turn's metadata with the structural facts the decision turn
    needs: the RUNNING tasks in THIS chat (so it can steer_task the right one
    instead of spawning a duplicate) and the originating message id (for idempotent
    steer delivery). P5-clean: surfaces state only; the agent picks the target by
    judgment among answer / steer_task / promote_chat_to_task / route_to_project."""
    md = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    swarm_intent = bool(md.get("force_plan"))
    addressable_here = _addressable_root_tasks(ctx, chat_id)
    running_here = [row for row in addressable_here if row.get("status") == "running"]
    project_id = str(md.get("project_id") or "").strip() or _project_id_for_registered_chat(
        ctx, chat_id,
    )
    is_main_lane = not bool(project_id)
    try:
        # Every non-Project owner transport is the Main lane.  External transports
        # commonly use a real provider chat id rather than Web's numeric ``1``;
        # keying this decision to ``chat_id == 1`` made their canonical router see
        # neither Projects nor globally addressable roots.
        main_manifest = _main_routing_manifest(ctx) if is_main_lane else {}
        if main_manifest and not (
            main_manifest.get("projects") or main_manifest.get("root_tasks")
        ):
            main_manifest = {}
    except Exception:
        log.warning("Unable to build Main routing manifest", exc_info=True)
        main_manifest = {"error": "routing_manifest_unavailable"} if is_main_lane else {}
    if not swarm_intent and not addressable_here and not client_message_id and not main_manifest:
        return task_metadata
    if addressable_here:
        md["current_chat"] = {
            "chat_id": int(chat_id or 0),
            "running_tasks": running_here,
            "addressable_root_tasks": addressable_here,
        }
    if main_manifest:
        md["main_routing_manifest"] = main_manifest
    if project_id:
        # Ground truth for a project-room "continue" decision (Q8-A): the thread's
        # most recent task result as a bounded typed projection. Without it the
        # router turn has only chat memory about where prior work lives.
        try:
            row = _latest_project_task_result(ctx, project_id)
            if row is not None:
                md["project_last_task_result"] = _task_result_ground_truth(row)
        except Exception:
            log.debug("project last-task-result projection failed", exc_info=True)
    if client_message_id:
        md["client_message_id"] = client_message_id
    option_roots = (
        list(main_manifest.get("root_tasks") or [])
        if is_main_lane and isinstance(main_manifest, dict)
        else addressable_here
    )
    manual_options = [] if swarm_intent else [
        {
            "action": "steer_task",
            "task_id": row["task_id"],
            "status": row["status"],
            "title": row.get("title") or row.get("objective"),
            "project_id": str(row.get("project_id") or ""),
        }
        for row in option_roots
        if isinstance(row, dict) and row.get("task_id")
    ]
    if not swarm_intent and is_main_lane and isinstance(main_manifest, dict):
        manual_options.extend({
            "action": "new_task_in_project",
            "project_id": str(row.get("project_id") or ""),
            "project_name": str(row.get("name") or row.get("project_id") or "Project"),
            "label": f"New task in {str(row.get('name') or 'Project')}",
        } for row in list(main_manifest.get("projects") or []) if isinstance(row, dict))
    elif project_id and not swarm_intent:
        manual_options.append({
            "action": "new_task_in_project",
            "project_id": project_id,
            "label": "New task in Project",
        })
    routing_contract = {
        "llm_first": True,
        "source_lane": "main" if is_main_lane else "project",
        "valid_actions": (
            (["promote_chat_to_task", "route_to_project"] if is_main_lane else ["promote_chat_to_task"])
            if swarm_intent else
            [
                "answer_inline", "steer_task", "promote_chat_to_task", "route_to_project",
                "needs_manual_target",
            ]
        ),
        "on_uncertain_or_invalid_target": (
            "promote_chat_to_task" if swarm_intent else "needs_manual_target"
        ),
        "manual_options": manual_options,
    }
    if not swarm_intent:
        routing_contract["manual_target_tool"] = {"name": "route_to_project", "project_id": ""}
    md["routing_contract"] = routing_contract
    return md


def _scoped_task_metadata(project_id: str, task_metadata: Any) -> Any:
    """Bind a chat frame's task_metadata to the thread's project via chat_id (the
    SSOT). A registered project chat scopes to its OWN project, overriding any
    client-supplied project_id; a non-project chat DROPS an untrusted client
    project_id (work is scoped to a project only via the promote_chat_to_task tool,
    never a raw ws frame). Prevents a stale/malformed frame (chat_id A + project_id
    B) from rendering in A while loading/writing project B's memory."""
    if project_id:
        return {**(task_metadata or {}), "project_id": project_id}
    if task_metadata and task_metadata.get("project_id"):
        return {k: v for k, v in task_metadata.items() if k != "project_id"}
    return task_metadata


def _owner_binding_chat_id(ctx: Any, chat_id: int, is_external_transport: bool) -> int:
    """The owner's canonical chat for owner-targeted notices (restart, supervisor
    death, consciousness). External transports bind to their own chat; a WEB owner
    always binds to MAIN (1), never a project panel — so if the first post-reset
    web message lands in a project room, owner notices still reach main."""
    if not is_external_transport and _project_id_for_registered_chat(ctx, chat_id):
        return 1
    try:
        return int(chat_id or 0)
    except (TypeError, ValueError):
        return 0


def _project_id_for_registered_chat(ctx: Any, chat_id: int) -> str:
    """Return the registered project id for a project chat_id, else ``""``.

    NOT an isolation gate (full project awareness, v6.32.0): the one mind notices
    EVERY human message via inject_observation, project rooms included. This just
    classifies a chat as a project thread so the message is scoped to that project
    (task_metadata.project_id) and routed to its panel. This active-only lookup is
    paired with ``_reserved_project_for_chat`` for deleting/tombstoned IDs, so a
    reserved chat cannot be resurrected through ordinary routing.
    """
    try:
        from ouroboros.projects_registry import list_projects

        cid = int(chat_id or 0)
        for project in list_projects(ctx.DRIVE_ROOT):
            try:
                if int(project.get("chat_id") or 0) == cid:
                    return str(project.get("id") or "").strip()
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("Project chat_id lookup failed", exc_info=True)
    return ""


def _reserved_project_for_chat(ctx: Any, chat_id: int) -> Dict[str, Any]:
    try:
        from ouroboros.projects_registry import list_reserved_projects

        cid = int(chat_id or 0)
        for project in list_reserved_projects(ctx.DRIVE_ROOT):
            try:
                if int(project.get("chat_id") or 0) == cid:
                    return dict(project)
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("Reserved Project chat lookup failed", exc_info=True)
    return {}
