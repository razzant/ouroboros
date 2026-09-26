"""Bounded facts one owner turn is allowed to address.

Projections only: which root tasks a chat can steer, what a project's last
result says about where its work lives, what the Main lane can see, and how a
chat maps to a project. Nothing here delivers a message or picks a target —
that judgment belongs to the decision turn (BIBLE P5).
"""

from __future__ import annotations

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


def _active_direct_roots(ctx: Any) -> list:
    """All addressable native actors, without constructing agents or queue state."""
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.workers import direct_chat_turn

    roots = []
    for entry in get_direct_activity_registry().actors():
        lock = getattr(entry.actor, "_owner_message_admission_lock", None)
        if lock is None:
            continue
        with lock:
            turn = direct_chat_turn(entry.activity_id)
            if turn is not None:
                roots.append({
                    "task_id": turn["id"], "status": "running",
                    "title": _clip_marked(turn.get("title"), 120),
                    "objective": _clip_marked(turn.get("text"), 600),
                    "project_id": turn["project_id"], "chat_id": turn["chat_id"],
                    "started_at": turn["_started_at"], "steerable": True, "direct_chat": True,
                })
    return roots


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
        # RUNNING can mean only paid post-work remains. A terminal result
        # cannot drain a new owner/peer message; don't suggest it as steerable.
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES
        from supervisor.queue import _task_drive_for_task

        if (load_task_result(_task_drive_for_task(task_obj, tid), tid) or {}).get("status") in SETTLED_STATUSES:
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
    for direct in _active_direct_roots(ctx):
        if direct["task_id"] not in seen and (
            chat_id is None or _task_belongs_to_chat(ctx, direct["task_id"], direct, chat_id)
        ):
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
    it never auto-chooses (BIBLE P5). Direct native roots are included;
    delegated subagents are not owner roots."""
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
    task_id = str(row.get("task_id") or row.get("id") or "")
    human_label = _clip_marked(
        row.get("title") or row.get("objective") or row.get("description") or task_id, 120,
    )
    out = {
        "task_id": task_id,
        "status": str(row.get("status") or ""),
        "title": _clip_marked(row.get("title"), 120),
        "objective": _clip_marked(row.get("objective") or row.get("description"), 300),
        "project_id": str(row.get("project_id") or ""),
        "reason_code": str(row.get("reason_code") or ""),
        "workspace_root": str(row.get("workspace_root") or ""),
        "workspace_mode": str(row.get("workspace_mode") or ""),
        "artifact_status": str(bundle.get("status") or row.get("artifact_status") or ""),
        "artifact_refs": [
            str(item.get("path") or item.get("name") or "")
            for item in artifacts[:8] if isinstance(item, dict)
        ],
        "authority_source": {
            "kind": "task_result",
            "task_id": task_id,
            "human_label": human_label,
            "tool": "get_task_result",
            "arguments": {"task_id": task_id, "include_authority": True},
        },
    }
    if git:
        out["workspace_git_at_start"] = {
            "head": str(git.get("head") or ""),
            "branch": str(git.get("branch") or ""),
            "dirty": bool(git.get("dirty")),
        }
    origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
    if origin:
        # WHY this result stopped, in the three scalars a continuation decision
        # needs; the full origin (actor, request id, observation) stays in the row.
        out["cancel_origin"] = {
            key: str(origin.get(key) or "")
            for key in ("reason", "source", "requested_at") if origin.get(key)
        }
    return out


def _is_child_result(facts: Dict[str, Any]) -> bool:
    """A result that is NOT an owner root: it has a parent, or the subagent role.

    ONE predicate for every reader of that fact - the manifest window that skips
    children (owner decision batch 3, answer 6b=A), the pointer stamp that only a
    root moves, and the promote receipt, which continues a named child with its
    root disclosed. Reads a memoized fact row or a full result row.
    """
    return bool(str(facts.get("parent_task_id") or "").strip()) or str(
        facts.get("delegation_role") or "") == "subagent"


def _recent_root_results(ctx: Any, project_id: str = "") -> tuple:
    """``(rows, omissions)``: the newest ROOT results a routing turn may continue,
    optionally narrowed to ONE project.

    The one producer behind both routing manifests - Main offers every lane's
    roots, a project room offers its own. Only the owner's ROOT results are
    offered (owner decision batch 3, answer 6b=A): a swarm wave's children are the
    newest results of ANY kind, so they evicted the owner's own roots from this
    window - which is how a root the same actor had just read stopped being
    offerable. The facts are already memoized, so both the filter and the count
    cost no extra read. The count runs over the WHOLE candidate list, not inside
    the capped loop: children older than the last shown root are skipped just the
    same, and counting them only until the cap reported zero while folding them
    into the cap's own number.
    """
    from ouroboros.gateway.task_list_scan import raw_result_facts
    from ouroboros.runtime_limits import get_routing_manifest_result_rows
    from ouroboros.task_results import load_task_result, task_results_dir

    results_error = ""
    try:
        facts, unreadable = raw_result_facts(task_results_dir(ctx.DRIVE_ROOT, create=False))
    except OSError as exc:
        facts, unreadable = {}, ["result_directory_unreadable"]
        results_error = f"result_directory_unreadable: {exc}"
    ordered = sorted(
        facts, key=lambda name: facts[name]["ts"] or facts[name]["updated_at"], reverse=True,
    )
    pool = [name for name in ordered if not project_id or facts[name]["project_id"] == project_id]
    children = sum(
        1 for name in pool if not facts[name]["schema_refusal"] and _is_child_result(facts[name])
    )
    cap = get_routing_manifest_result_rows()
    finals: list = []
    for name in pool:
        if facts[name]["schema_refusal"] or _is_child_result(facts[name]):
            continue
        row = load_task_result(ctx.DRIVE_ROOT, pathlib.Path(name).stem)
        if row is not None:
            finals.append(_task_result_ground_truth(row))
        if len(finals) == cap:
            break
    return finals, {
        # Kept meaning: results cut by the row cap. The children skipped above are
        # a DIFFERENT omission and are counted as such, never folded in here.
        "final_results": None if unreadable else max(0, len(pool) - children - len(finals)),
        "final_results_error": results_error,
        "children": None if unreadable else children,
    }


def _cancel_state_facts(ctx: Any, task_id: str) -> Dict[str, Any]:
    """The durable cancel intent standing over one LIVE root, or ``{}``: the typed
    public projection (``cancel_state``/``cancel_reason``/``stop_policy``), never a
    body. A settled result carries its own ``cancel_origin`` instead."""
    if not task_id:
        return {}
    try:
        from ouroboros.cancel_intents import cancel_state_fields

        return cancel_state_fields(ctx.DRIVE_ROOT, task_id)
    except Exception:
        log.debug("cancel state unreadable for %s", task_id, exc_info=True)
        return {}


def _project_routing_manifest(ctx: Any, project_id: str) -> Dict[str, Any]:
    """The room's bounded HINT for a "continue this work" decision: the project's
    recent ROOT results and the roots still live in it, each with the small typed
    facts that separate the two choices - a settled root is promote's predecessor,
    a live one is ``steer_task``.

    A hint, never the door: promote's predicate admits any settled result, listed
    or not, of any project (ch. 10), so this window may be bounded without
    deciding what the room can continue. Until it existed a room saw exactly ONE
    candidate, the registry pointer, so a room whose pointer had moved could not
    name its own interrupted root at all.
    """
    finals, omissions = _recent_root_results(ctx, project_id)
    active = [
        {**row, **_cancel_state_facts(ctx, str(row.get("task_id") or ""))}
        for row in _addressable_root_tasks(ctx, None)
        if str(row.get("project_id") or "") == project_id
    ]
    return {
        "final_results": finals,
        "active_roots": active[:40],
        "omissions": {**omissions, "active_roots": max(0, len(active) - 40)},
    }


def _not_a_root_result(row: Dict[str, Any]) -> bool:
    """A row that is never "the project's last result": a child's result, or a
    promote's emitted stub (an admission still pending, no result at all)."""
    from ouroboros.routing_wait import is_emitted_admission_stub

    return _is_child_result(row) or is_emitted_admission_stub(row)


def _latest_project_task_result(ctx: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Newest ROOT task result bound to ``project_id`` (a child's is never the room's
    last-result pointer: the hint offers roots only, ``_is_child_result``) WITHOUT replaying the whole
    store (DEVELOPMENT "Projection over replay"). The registry row's durable
    ``last_task_result_id`` pointer (stamped at project-task finalization) is
    read FIRST — one direct file fetch, immune to how many newer foreign
    results exist. Only when the pointer is absent or stale (missing/
    unparseable/foreign file) does the fallback run: the bounded newest-64
    mtime scan, then — for pre-pointer projects only — a disclosed full scan
    of the store (the lazy self-heal for rows finalized before the pointer
    existed; with zero matching results nothing is written back, so it repeats
    per lookup until a matching result exists). The order is TOTAL: newest
    mtime first with the file name as the stable tie-break, and — because
    results finalized within one clock tick share an mtime — the first match's
    whole equal-mtime group is read to its end (across the 64-entry window,
    which bounds the SEARCH, not the group) and the durable ``ts`` decides
    inside it (``updated_at`` when ``ts`` is absent, the task id last). A file
    whose stat or JSON could not be read is answered around best-effort for
    the call but blocks the pointer write-back (it might be the newer result).
    Only the ABSENT-pointer case
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
            if not _not_a_root_result(pointed):
                return pointed
            pointer = ""  # a child-stamped pointer is provably wrong, not in flight: heal it
        log.debug(
            "project last-task-result pointer for %r is stale (%s); "
            "falling back to the bounded scan", project_id, pointer,
        )

    def _stamp(path: Any) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except OSError:
            return None  # unreadable stat: ordered last, member of no tie group

    def _order(item: Dict[str, Any]) -> tuple:
        # Inside an equal-mtime group the DURABLE `ts` decides (`updated_at`
        # stands in when `ts` is absent); the task id is the deterministic last
        # resort for equal keys (a stable choice, not a semantic one).
        return (str(item.get("ts") or item.get("updated_at") or ""), str(item.get("task_id") or item.get("id") or ""))

    stamped = [(_stamp(path), path) for path in task_results_dir(ctx.DRIVE_ROOT, create=False).glob("*.json")]
    # A file whose stat or JSON cannot be read might be this project's newer
    # result: the scan still answers best-effort for THIS call, but such
    # uncertainty must never be frozen into the durable pointer.
    uncertain = any(mtime is None for mtime, _ in stamped)
    # Newest first; unreadable last; the name keeps equal mtimes contiguous
    # and the whole order deterministic.
    stamped.sort(key=lambda item: (item[0] is None, -(item[0] or 0.0), item[1].name))
    row = None
    for index, (mtime, path) in enumerate(stamped):
        if index == 64:
            log.info(
                "project last-task-result: %r missed the bounded scan; running the "
                "full-store self-heal scan (%d files)", project_id, len(stamped),
            )
        candidate = read_json_dict(path)
        if candidate is None:
            uncertain = True
            continue
        if str(candidate.get("project_id") or "") != project_id or _not_a_root_result(candidate):
            continue
        row = candidate
        # The match's whole equal-mtime group is read to its end — across the
        # 64-entry window too, which bounds the SEARCH and never cuts a group
        # whose order the mtime cannot settle.
        for tied_mtime, tied in stamped[index + 1:]:
            if mtime is None or tied_mtime != mtime:
                break
            other = read_json_dict(tied)
            if other is None:
                uncertain = True
            elif (str(other.get("project_id") or "") == project_id and not _not_a_root_result(other)
                  and _order(other) > _order(row)):
                row = other
        break
    if row is not None and not pointer and not uncertain:
        try:
            update_project(ctx.DRIVE_ROOT, project_id, last_task_result_id=str(
                row.get("task_id") or row.get("id") or ""))
        except Exception:
            log.debug("project last-task-result pointer write-back failed", exc_info=True)
    return row


def _main_routing_manifest(ctx: Any) -> Dict[str, Any]:
    """Bounded canonical facts for one Main-chat LLM routing decision."""
    from ouroboros.gateway._helpers import read_rotated_jsonl_entries
    from ouroboros.projects_registry import list_projects

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
    finals, result_omissions = _recent_root_results(ctx)

    dialogue_rows: list = []
    root = pathlib.Path(ctx.DRIVE_ROOT)
    rows, gaps = read_rotated_jsonl_entries(
        root / "logs" / "chat.jsonl", root / "archive", "chat", 20,
        lambda row: bool(str(row.get("text") or "").strip()),
        max_archives=2, include_gaps=True,
    )
    for row in rows:
        text = str(row.get("text") or "").strip()
        if text:
            dialogue_rows.append({
                "ts": str(row.get("ts") or ""),
                "direction": str(row.get("direction") or ""),
                "chat_id": row.get("chat_id", 1),
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
            **result_omissions,
            # A bounded read cannot count bytes/rows it deliberately did not
            # visit. The exact historical messages remain available by id.
            "dialogue_rows": None,
            "dialogue_source": "chat_history (canonical chat and archives)",
            "dialogue_gaps": sorted(gaps),
        },
    }


def _decision_turn_metadata(ctx: Any, chat_id: int, client_message_id: str, task_metadata: Any) -> Any:
    """Enrich a chat turn's metadata with the structural facts the decision turn
    needs: the RUNNING tasks in THIS chat (so it can steer_task the right one
    instead of spawning a duplicate) and the originating message id (for idempotent
    steer delivery). P5-clean: surfaces state only; the agent picks the target by
    judgment among answer / steer_task / promote_chat_to_task / route_to_project."""
    md = dict(task_metadata) if isinstance(task_metadata, dict) else {}
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
    if not addressable_here and not client_message_id and not main_manifest:
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
        try:
            md["project_routing_manifest"] = _project_routing_manifest(ctx, project_id)
        except Exception:
            log.warning("Unable to build the project routing manifest", exc_info=True)
    if client_message_id:
        md["client_message_id"] = client_message_id
    option_roots = (
        list(main_manifest.get("root_tasks") or [])
        if is_main_lane and isinstance(main_manifest, dict)
        else addressable_here
    )
    manual_options = [
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
    if is_main_lane and isinstance(main_manifest, dict):
        manual_options.extend({
            "action": "new_task_in_project",
            "project_id": str(row.get("project_id") or ""),
            "project_name": str(row.get("name") or row.get("project_id") or "Project"),
            "label": f"New task in {str(row.get('name') or 'Project')}",
        } for row in list(main_manifest.get("projects") or []) if isinstance(row, dict))
    elif project_id:
        manual_options.append({
            "action": "new_task_in_project",
            "project_id": project_id,
            "label": "New task in Project",
        })
    routing_contract = {
        "llm_first": True,
        "source_lane": "main" if is_main_lane else "project",
        "valid_actions": [
            "answer_inline", "steer_task", "promote_chat_to_task", "route_to_project",
            "needs_manual_target",
        ],
        "on_uncertain_or_invalid_target": "needs_manual_target",
        "manual_options": manual_options,
    }
    routing_contract["manual_target_tool"] = {"name": "route_to_project", "project_id": ""}
    receipt = _message_routing_receipt(ctx, client_message_id)
    if receipt:
        # DISCLOSURE, not a gate (owner decision B5=A): one owner message became a
        # task and was then steered into three more live roots, each paying its own
        # review wave, because the deciding turn was never told a receipt already
        # existed. The choice stays with the model - no host ban on a second root.
        routing_contract["message_routing_receipt"] = receipt
    md["routing_contract"] = routing_contract
    return md


def main_lane_routing_metadata(ctx: Any, chat_id: int) -> Dict[str, Any]:
    """The Main-lane routing facts for a turn NOBODY typed (a consciousness wake-up).

    Exactly what an owner turn in the same chat is handed — the Main routing manifest
    and this chat's addressable roots — minus what is bound to an owner message (there
    is none). One seam over the owner path, so a wake can never drift from what the
    host shows an owner turn: the manifest is the hint both decide from, while the
    door judges the named result itself, listed or not.
    """
    facts = _decision_turn_metadata(ctx, int(chat_id or 0), "", {})
    return dict(facts) if isinstance(facts, dict) else {}


def _message_routing_receipt(ctx: Any, client_message_id: str) -> Dict[str, Any]:
    """The existing routing receipt for THIS owner message, or {} when there is none.

    Read from the annotation the routing rail already writes, so no new store and no
    new reader: the decision turn simply sees what was already decided for the same
    message. Fail-soft - a missing or torn annotations file leaves the turn exactly
    as it was.
    """
    if not client_message_id:
        return {}
    try:
        from ouroboros.project_dialogue import latest_chat_annotations

        row = latest_chat_annotations(ctx.DRIVE_ROOT).get(str(client_message_id)) or {}
    except Exception:
        log.debug("message routing receipt lookup failed", exc_info=True)
        return {}
    if not row:
        return {}
    return {
        "action": str(row.get("action") or ""),
        "target": str(row.get("target") or ""),
        "target_label": str(row.get("target_label") or ""),
        "status": str(row.get("status") or ""),
        "ts": str(row.get("ts") or ""),
        "project_id": str(row.get("project_id") or ""),
    }


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

    NOT an isolation gate (full project awareness, v6.32.0): the one mind sees
    EVERY human message in its own Main context, project rooms included. This just
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
