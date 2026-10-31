"""Bounded facts one owner turn is allowed to address.

Projections only: which root tasks a chat can steer, what a project's last
result says about where its work lives, what the Main lane can see, and how a
chat maps to a project. Nothing here delivers a message or picks a target —
that judgment belongs to the decision turn (BIBLE P5).

Main's manifest lists each project's registry ``working_dir`` and addresses of
the last 20 spoken rows; the memory view holds their text. Project rooms and Main read recent root results and live roots as
bounded typed projections, never raw result text; a project's last result is
found through the registry row's ``last_task_result_id`` pointer (stamped by
roots only), with a bounded scan as fallback.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ouroboros.server_process import log


@dataclass(frozen=True)
class RootSnapshot:
    """One owner message's addressable roots, read once and handed on by value.

    ``pairs`` are ``(manifest row, chat ids it belongs to)``: the unsettled RUNNING and PENDING
    roots in queue order, then the live direct turns. ``gap`` names a read that failed: the
    roots are then unknown, which the routing metadata says, never an empty room.
    """

    pairs: tuple = ()
    gap: str = ""

    def rows(self, chat_id: Optional[int] = None) -> list:
        """Copies of the rows one chat (``None``: any) may address; a reading direct turn
        replaces its own queue row."""
        wanted, out, seen = None if chat_id is None else int(chat_id or 0), [], set()
        for row, chats in self.pairs:
            tid = row["task_id"]
            if row.get("direct_chat") and tid in seen and row.get("room_reader"):
                out, seen = [kept for kept in out if kept["task_id"] != tid], seen - {tid}
            if tid in seen or (wanted is not None and wanted not in chats):
                continue
            out.append(dict(row))
            if not row.get("direct_chat"):
                seen.add(tid)
        return out


def _task_chats(ctx: Any, task_id: str, task: Dict[str, Any]) -> frozenset:
    """A root's own ``chat_id`` and its bound Project room (0 when unbound); an unreadable fact adds none."""
    chats = set()
    try:
        chats.add(int(task.get("chat_id") or 0))
    except (TypeError, ValueError):
        pass
    try:
        from ouroboros.projects_registry import project_chat_for_task

        chats.add(int(project_chat_for_task(ctx.DRIVE_ROOT, task_id) or 0))
    except Exception:
        pass
    return frozenset(chats)


def _late_input_author(task: Dict[str, Any]) -> Dict[str, str]:
    """``{"late_input_of": author}`` for a head a late answer, or a press on a finished author's
    card, started (typed provenance), else empty."""
    from ouroboros.card_store import late_press_ref
    from ouroboros.owner_quiz import late_answer_ref

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    ref = late_answer_ref(metadata.get("late_answer")) or late_press_ref(metadata.get("card_press"))
    return {"late_input_of": ref["task_id"]} if ref else {}


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
                from ouroboros.room_reader import is_reading
                reader = is_reading(ctx.DRIVE_ROOT, turn["id"], turn, running=True)
                roots.append({
                    "task_id": turn["id"], "status": "running",
                    "title": _clip_marked(turn.get("title"), 120),
                    "objective": _clip_marked(turn.get("text"), 600),
                    "project_id": turn["project_id"], "chat_id": turn["chat_id"],
                    "started_at": turn["_started_at"], "steerable": True, "direct_chat": True, "room_reader": reader,
                    **_late_input_author(turn),
                })
    return roots


def addressable_roots(ctx: Any) -> RootSnapshot:
    """The roots one owner message may address, without choosing a target (``RootSnapshot``).

    One typed read gives every root's status and own drive; no result body is decoded. A root
    its actor's drive reports settled is left out (a split root's child drive: copyback can
    lag): RUNNING can mean only paid post-work remains, and a terminal result cannot drain a
    new owner or peer message, so it is never suggested as steerable. The running roots'
    owner-Pause fences are read in one statement per store and handed to each reader check.
    """
    try:
        from ouroboros.room_reader import is_reading
        from ouroboros.store_hot_facts import snapshot as hot_facts
        from ouroboros.task_status import SETTLED_STATUSES
        from ouroboros.task_store import task_drive_status

        queued = [*((tid, row.get("task") if isinstance(row.get("task"), dict) else row, "running",
                     row.get("started_at")) for tid, row in list(getattr(ctx, "RUNNING", {}).items())
                    if isinstance(row, dict)),
                  *((row.get("id"), row, "pending", row.get("queued_at"))
                    for row in list(getattr(ctx, "PENDING", []) or []) if isinstance(row, dict))]
        roots = [(str(tid or "").strip(), task, status, started) for tid, task, status, started in queued
                 if str(tid or "").strip() and isinstance(task, dict) and task.get("delegation_role") != "subagent"]
        canonical = pathlib.Path(ctx.DRIVE_ROOT) if roots else None  # no root, no store read
        stored = task_drive_status(canonical, [root[0] for root in roots])
        drives = [pathlib.Path(str(task.get("child_drive_root") or task.get("drive_root") or "").strip()
                               or stored.get(tid, ("", ""))[1] or canonical) for tid, task, *_ in roots]
        statuses = {(canonical, tid): fact[0] for tid, fact in stored.items()}
        for drive in set(drives) - {canonical}:
            ids = [root[0] for root, actor in zip(roots, drives) if actor == drive]
            statuses.update(((drive, tid), fact[0]) for tid, fact in task_drive_status(drive, ids).items())
        live = [(root, drive) for root, drive in zip(roots, drives)
                if statuses.get((drive, root[0]), "") not in SETTLED_STATUSES]
        pairs = []
        with hot_facts([(drive, str(task.get("root_task_id") or tid))  # a store that cannot answer raises below
                        for (tid, task, status, _at), drive in live if status == "running"]):
            for (tid, task, status, started_at), drive in live:
                objective = str(task.get("objective") or task.get("description") or task.get("text") or "").strip()
                reader = is_reading(drive, tid, task, running=status == "running",
                                    fences=getattr(ctx, "BUDGET_ROOT_FENCES", None), drain_ended=False)
                pairs.append(({
                    "task_id": tid,
                    "status": status,
                    "title": _clip_marked(task.get("title"), 120),
                    "objective": _clip_marked(objective, 600),
                    "project_id": str(task.get("project_id") or ""),
                    "started_at": started_at,
                    "steerable": True, "room_reader": reader, **_late_input_author(task),
                }, _task_chats(ctx, tid, task)))
        pairs += [(row, _task_chats(ctx, row["task_id"], row)) for row in _active_direct_roots(ctx)]
        return RootSnapshot(tuple(pairs))
    except Exception as exc:
        log.warning("Addressable roots are unavailable to this routing read", exc_info=True)
        return RootSnapshot(gap=type(exc).__name__)


def _addressable_root_tasks(ctx: Any, chat_id: Optional[int] = None) -> list:
    """Compact RUNNING+PENDING owner-root manifest of one chat (``None``: every chat)."""
    return addressable_roots(ctx).rows(chat_id)


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
    pick a tell_task target by its own judgment — code only exposes the state,
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
    if isinstance(row.get("promotion_admission"), dict):
        from ouroboros.promotion_intent import promotion_intent_notice
        out.update(promotion_admission=dict(row["promotion_admission"]),
                   promotion_intent_available=bool(row.get("promotion_intent")),
                   admission_notice=promotion_intent_notice(task_id, row))
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


def _result_reference(facts: Dict[str, Any]) -> Dict[str, Any]:
    """A historical result's identity and exact reader, without its evidence body."""
    return {key: facts[key] for key in ("task_id", "status", "title", "authority_source")}


def _is_child_result(facts: Dict[str, Any]) -> bool:
    """A result that is NOT an owner root: it has a parent, or the subagent role.

    ONE predicate for every reader of that fact - the pointer stamp that only a
    root moves and the promote receipt, which continues a named child with its
    root disclosed; the manifest window that skips children (owner decision batch
    3, answer 6b=A) selects by the same fact over typed columns in the store
    (``task_store.ROOT_RESULT_WHERE``). Reads a memoized fact row or a full result row.
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
    offerable. Owner-message intake runs on the supervisor loop, so the store
    selects only the rows the window can offer (``newest_root_ids``): typed columns,
    bodies decoded only for the rows shown. The two omissions stay separate and
    truthful: a project room counts its own roots and children over its own index;
    Main's scope is the whole store, so its window proves only whether more roots
    exist (``0`` or ``None``, not counted) and leaves its children uncounted, like
    the dialogue rows it does not visit.
    """
    from ouroboros.runtime_limits import get_routing_manifest_result_rows
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import newest_root_ids, project_result_counts

    cap = get_routing_manifest_result_rows()
    try:
        ids = newest_root_ids(ctx.DRIVE_ROOT, cap + 1, project_id=project_id)
        counted = project_result_counts(ctx.DRIVE_ROOT, project_id) if project_id else None
    except StoreError as exc:
        return [], {"final_results": None, "final_results_error": f"result_store_unreadable: {exc}", "children": None}
    finals = [_task_result_ground_truth(row) for row in
              (load_task_result(ctx.DRIVE_ROOT, task_id) for task_id in ids[:cap]) if row is not None]
    roots, children = counted if counted is not None else (None, None)
    return finals, {
        # Kept meaning: results cut by the row cap. Skipped children are a DIFFERENT
        # omission, never folded in here.
        "final_results": max(0, roots - len(finals)) if roots is not None else (0 if len(ids) <= cap else None),
        "final_results_error": "",
        "children": children,
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


def _project_routing_manifest(ctx: Any, project_id: str, roots: Optional[RootSnapshot] = None) -> Dict[str, Any]:
    """The room's bounded HINT for a "continue this work" decision: the project's
    recent ROOT results and the roots still live in it, each with the small typed
    facts that separate the two choices - a settled root is promote's predecessor,
    a live one is ``tell_task``.

    A hint, never the door: promote's predicate admits any settled result, listed
    or not, of any project (ch. 10), so this window may be bounded without
    deciding what the room can continue. Until it existed a room saw exactly ONE
    candidate, the registry pointer, so a room whose pointer had moved could not
    name its own interrupted root at all.
    """
    finals, omissions = _recent_root_results(ctx, project_id)
    roots = roots if roots is not None else addressable_roots(ctx)
    active = [
        {**row, **_cancel_state_facts(ctx, str(row.get("task_id") or ""))}
        for row in roots.rows()
        if str(row.get("project_id") or "") == project_id
    ]
    return {
        "final_results": [_result_reference(row) for row in finals],
        "active_roots": active[:40],
        "omissions": {**omissions, "active_roots": None if roots.gap else max(0, len(active) - 40)},
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
    read FIRST — one indexed row read, immune to how many newer foreign
    results exist. Only when the pointer is absent or stale (missing/foreign
    row) does the fallback run: the project's root rows newest first by the
    durable ``ts`` (``newest_root_ids``: its own index, never every root), read
    in that order until the first that is not a promote's emitted stub. Only the
    ABSENT-pointer case writes the pointer back: a non-empty pointer that failed
    to resolve is usually a split-drive result in flight (finalization stamps
    the pointer before the canonical copy-back lands), so overwriting it from
    the scan would permanently regress it to an older result — serve the scan
    hit and let the pointer resolve itself."""
    from ouroboros.projects_registry import get_project, update_project
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import newest_root_ids

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

    row = None
    for task_id in newest_root_ids(ctx.DRIVE_ROOT, project_id=project_id):
        candidate = load_task_result(ctx.DRIVE_ROOT, task_id)
        if candidate is None or _not_a_root_result(candidate):
            continue
        row = candidate
        break
    if row is not None and not pointer:
        try:
            update_project(ctx.DRIVE_ROOT, project_id, last_task_result_id=str(
                row.get("task_id") or row.get("id") or ""))
        except Exception:
            log.debug("project last-task-result pointer write-back failed", exc_info=True)
    return row


def _main_routing_manifest(ctx: Any, roots: Optional[RootSnapshot] = None) -> Dict[str, Any]:
    """Routing identities and exact readers; the memory view owns spoken rows."""
    from ouroboros import dialogue_store as ds
    from ouroboros.chat_chain import row_address, format_address
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
    snapshot = roots if roots is not None else addressable_roots(ctx)
    roots = snapshot.rows()
    finals, result_omissions = _recent_root_results(ctx)

    dialogue_rows: list = []
    root = pathlib.Path(ctx.DRIVE_ROOT)
    upper = before = ds.stream_end(root, "chat")
    rows, gaps = [], set()
    while before and len(rows) < 20:
        page = ds.newest(root, "chat", before=before, limit=20)
        if not page:
            before = 0
            break
        rows.extend(entry for entry in reversed(page) if str(entry.row.get("text") or "").strip())
        before = page[0].seq
    coverage = ds.range_coverage(root, "chat", lower=rows[19].seq if len(rows) >= 20 else 0, upper=upper)
    if not coverage["imported"]:
        gaps.add("a segment could not be imported" if coverage["mismatch"] else "history is still importing")
    if coverage["gaps"]:
        gaps.add("recorded invalid intervals")
    rows = [entry.row for entry in reversed(rows[:20])]
    for row in rows:
        text = str(row.get("text") or "").strip()
        if text:
            dialogue_rows.append({
                "ts": str(row.get("ts") or ""),
                "direction": str(row.get("direction") or ""),
                "chat_id": row.get("chat_id", 1),
                "chars": len(text),
                "reader": {"tool": "memory_read", "arguments": {"room_id": str(row.get("chat_id", 1)),
                    "rows": True, "from": format_address(row_address(row)), "to": format_address(row_address(row))}},
                "task_id": str(row.get("task_id") or ""),
                "client_message_id": str(row.get("client_message_id") or ""),
            })
    dialogue = dialogue_rows[-20:]
    return {
        "projects": projects[:40],
        "root_tasks": roots[:40],
        "final_results": [_result_reference(row) for row in finals],
        "recent_canonical_dialogue": dialogue,
        "omissions": {
            "projects": max(0, len(projects) - 40),
            "root_tasks": None if snapshot.gap else max(0, len(roots) - 40),
            **result_omissions,
            # A bounded read cannot count bytes/rows it deliberately did not
            # visit. The exact historical messages remain available by id.
            "dialogue_rows": None,
            "dialogue_source": "chat_history (state/dialog.sqlite)",
            "dialogue_gaps": sorted(gaps),
        },
    }


def _decision_turn_metadata(ctx: Any, chat_id: int, client_message_id: str, task_metadata: Any,
                            roots: Optional[RootSnapshot] = None) -> Any:
    """Enrich a chat turn's metadata with the structural facts the decision turn
    needs: the RUNNING tasks in THIS chat (so it can tell_task the right one
    instead of spawning a duplicate) and the originating message id (for idempotent
    steer delivery). P5-clean: surfaces state only; the agent picks the target by
    judgment among answer / tell_task / start_task / move_to_project. ``roots``: the
    message's one ``addressable_roots`` snapshot, read here when not handed down."""
    md = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    roots = roots if roots is not None else addressable_roots(ctx)
    addressable_here = roots.rows(chat_id)
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
        main_manifest = _main_routing_manifest(ctx, roots) if is_main_lane else {}
        if main_manifest and not (
            main_manifest.get("projects") or main_manifest.get("root_tasks")
        ):
            main_manifest = {}
    except Exception:
        log.warning("Unable to build Main routing manifest", exc_info=True)
        main_manifest = {"error": "routing_manifest_unavailable"} if is_main_lane else {}
    if not addressable_here and not client_message_id and not main_manifest and not roots.gap:
        return md
    if addressable_here:
        md["current_chat"] = {
            "chat_id": int(chat_id or 0),
            "running_tasks": running_here,
            "addressable_root_tasks": addressable_here,
        }
    if main_manifest:
        md["main_routing_manifest"] = main_manifest
    if project_id:
        # Name the thread's latest result and its exact reader; prior workspace
        # and outcome evidence remains in that result instead of riding inline.
        try:
            row = _latest_project_task_result(ctx, project_id)
            if row is not None:
                md["project_last_task_result"] = _result_reference(_task_result_ground_truth(row))
        except Exception:
            log.debug("project last-task-result projection failed", exc_info=True)
        try:
            md["project_routing_manifest"] = _project_routing_manifest(ctx, project_id, roots)
        except Exception:
            log.warning("Unable to build the project routing manifest", exc_info=True)
    if client_message_id:
        md["client_message_id"] = client_message_id
    routing_contract = {
        "llm_first": True,
        "source_lane": "main" if is_main_lane else "project",
        "valid_actions": [
            "answer_inline", "tell_task", "start_task", "move_to_project",
        ],
        # Unknown, never none: the roots of this chat and of Main could not be read now.
        **({"live_roots_unavailable": roots.gap} if roots.gap else {}),
    }
    receipt = _message_routing_receipt(ctx, client_message_id)
    if receipt:
        # DISCLOSURE, not a gate (owner decision B5=A): one owner message became a
        # task and was then steered into three more live roots, each paying its own
        # review wave, because the deciding turn was never told a receipt already
        # existed. The choice stays with the model - no host ban on a second root.
        routing_contract["message_routing_receipt"] = receipt
        acts = _message_routing_acts(ctx, client_message_id)
        if len(acts) > 1:
            # The latest row hides earlier acts on the same message (a promote, then
            # a steer): each act keeps its own receipt, read, never inferred.
            routing_contract["message_routing_acts"] = acts
            routing_contract["message_routing_acts_note"] = (
                "Recorded routing acts already taken for THIS owner message, oldest first. "
                "Each receipt identifies the work and its recorded outcome.")
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
    message (a promote's or route's status from its admission, as every reader shows
    it: ``routing_wait.routing_act_fields``). Fail-soft - a missing or torn
    annotations file leaves the turn exactly as it was.
    """
    if not client_message_id:
        return {}
    try:
        from ouroboros.project_dialogue import latest_chat_annotations
        from ouroboros.routing_wait import routing_act_fields

        row = latest_chat_annotations(ctx.DRIVE_ROOT).get(str(client_message_id)) or {}
        return routing_act_fields(pathlib.Path(ctx.DRIVE_ROOT), str(client_message_id), row) if row else {}
    except Exception:
        log.debug("message routing receipt lookup failed", exc_info=True)
        return {}


def _message_routing_acts(ctx: Any, client_message_id: str) -> list:
    """Every routing act recorded for THIS owner message (latest row per act), oldest first.

    The routing rail already keeps one receipt per (message, routing token); this
    only reads it. Acts keyed to a synthetic ``agent-steer:*`` id or to another
    task's id carry no link to this message and stay unlisted (a producer gap).
    """
    try:
        from ouroboros.project_dialogue import _ANNOTATIONS_NAME, _latest_annotations_by_token
        from ouroboros.routing_wait import routing_act_fields

        root = pathlib.Path(ctx.DRIVE_ROOT)
        rows = [row for (message_id, _token), row in _latest_annotations_by_token(
            root / "logs" / _ANNOTATIONS_NAME).items() if message_id == str(client_message_id)]
        return [routing_act_fields(root, str(client_message_id), row)
                for row in sorted(rows, key=lambda row: str(row.get("ts") or ""))]
    except Exception:
        log.debug("message routing acts lookup failed", exc_info=True)
        return []


def _scoped_task_metadata(project_id: str, task_metadata: Any) -> Any:
    """Bind a chat frame's task_metadata to the thread's project via chat_id (the
    SSOT). A registered project chat scopes to its OWN project, overriding any
    client-supplied project_id; a non-project chat DROPS an untrusted client
    project_id (work placement uses durable binding, never a raw ws frame).
    Prevents a stale/malformed frame (chat_id A + project_id
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


def _main_lane_chat(chat_id: int) -> bool:
    """Host-attested absence of a Project room, answered without the registry.

    Ids below the Project floor are Main, hidden or A2A. A transport id shares
    the numeric range, so only the positively bound external owner slot (written
    by that transport's own slash command) proves its chat is the owner's Main;
    any other id at or above the floor still needs the strict registry read.
    """
    from ouroboros.contracts.chat_id_policy import is_project_chat_id

    cid = int(chat_id or 0)
    if not is_project_chat_id(cid):
        return True
    try:
        from supervisor.state import control_value, load_state

        known, bound = control_value(load_state(), "owner_external_chat_id")
    except Exception:
        return False  # an unreadable binding proves nothing
    return known and type(bound) is int and bound == cid


def _reserved_project_for_chat(ctx: Any, chat_id: int) -> Dict[str, Any]:
    """Execution routing requires positive absence, not an unavailable display lens."""
    from ouroboros.project_admission import reserved_project_for_chat

    cid = int(chat_id or 0)
    if _main_lane_chat(cid):
        return {}  # Host-attested: this chat cannot belong to a Project room.
    return reserved_project_for_chat(ctx.DRIVE_ROOT, cid)
