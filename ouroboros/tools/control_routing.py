"""Start separate work, place the current work, and deliver messages to live work.

Tools state an intent; the supervisor owns admission and its durable receipt.
"""

from __future__ import annotations

import logging
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ouroboros.dialogue_provenance import presence_root_carrier
from ouroboros.tools.control_events import (
    _emit_and_wait_for_routing,
    _promotion_pool_disabled_from_snapshot,
)
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import completed_local_read
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)




def _root_effort_arg(value: Any, tool_name: str) -> tuple[Dict[str, str], str, str]:
    """``(event fields, refusal, note)`` for the new root's explicit starting effort.

    Omitted (``None``) leaves the field absent, so the root starts at the owner's
    recommended level; a supplied value must name a tier, before any effect. Outside
    Cyber Pro a root Ouroboros creates itself is not pinned by that value either: the
    field stays absent and the note says so (owner ingress — the API, the CLI, owner
    schedules, Continue — keeps pinning); in Cyber Pro it is applied.
    """
    if value is None or value == "":
        return {}, "", ""
    from ouroboros.runtime_mode_policy import effort_range_binds
    from ouroboros.settings_scales import effort_range, requested_effort
    from ouroboros.tools.arg_feedback import ignored_argument_note

    try:
        tier = requested_effort(value)
    except ValueError as exc:
        return {}, f"⚠️ TOOL_ARG_ERROR ({tool_name}): {exc}", ""
    if effort_range_binds():
        return {}, "", ignored_argument_note(
            "reasoning_effort", tier,
            f"outside Cyber Pro a root I create starts at my human's recommended level "
            f"({effort_range()['recommended']}); deeper thinking is delegated with schedule_subagent(effort=...)")
    return {"reasoning_effort": tier}, "", ""


def _attach_origin_from_metadata(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Copy the ingress-captured owner-message origin (ref + full text) onto a
    promote-shaped event BY VALUE. The host built the ref at chat admission;
    producers never re-derive identity from content (DEVELOPMENT.md
    anti-pattern: content-derived identity for host-minted records)."""
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(metadata, dict):
        return
    ref = metadata.get("origin_message_ref")
    if isinstance(ref, dict) and ref:
        evt["source_ref"] = dict(ref)
        text = metadata.get("origin_message_text")
        if isinstance(text, str) and text:
            evt["source_text"] = text
    elif metadata.get("origin_suppressed"):
        evt["origin_suppressed"] = True


def _attach_drafted_objective(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Keep the routed objective's author separate from the ingress owner corpus.

    A routing run with neither owner-door stamp opens the new root with this drafted
    objective and no owner message. Only this producer knows that, so it records
    ``origin_message_absent``: a later reader cannot tell an absent stamp from a lost
    one, and a suppressed stamp is owner input that the door did not log.

    The corpus is persisted with the start (its receipt, the queue snapshot, the running
    record), so each row is the retained owner corpus's text (``_owner_content_projection``):
    the owner's words verbatim and an image as its reference beside the path of its
    retained original, never its base64.
    """
    from ouroboros.review_evidence_sections import _owner_content_projection

    evt["objective_author"] = {"kind": "task", "task_id": str(getattr(ctx, "task_id", "") or "")}
    if not (evt.get("source_ref") or evt.get("origin_suppressed")):
        evt["origin_message_absent"] = True
    owner_rows = [dict(row) for row in (getattr(ctx, "_owner_directives", None) or [])
                  if isinstance(row, dict) and row.get("source") in {
                      "owner_mailbox", "owner_quiz_answer", "origin_message", "owner_corpus", "direct_incoming"}]
    if evt.get("source_text") and not any(row.get("source") == "origin_message" for row in owner_rows):
        owner_rows.insert(0, {"source": "origin_message", "content": evt["source_text"]})
    elif not evt.get("source_text"):
        # A suppressed (never-logged) origin carries no text; the owner's words
        # then live only in this run's first row, and ONLY under the host's
        # owner-ingress stamp (``initial_user``) — an unstamped first turn
        # (``initial_text``) is never laundered into owner authority.
        owner_rows[:0] = [dict(row) for row in (getattr(ctx, "_owner_directives", None) or [])
                          if isinstance(row, dict) and row.get("source") == "initial_user"]
    rows = [{**row, "content": _owner_content_projection(row.get("content"))} for row in owner_rows]
    evt["owner_corpus"] = [row for row in rows if row["content"].strip()]


def _inherited_project_scope(ctx: ToolContext) -> str:
    """Only the caller's own room supplies a default for separate new work."""
    from ouroboros.projects_registry import project_id_for_task

    return str(project_id_for_task(canonical_data_root(ctx), str(getattr(ctx, "task_id", "") or ""), strict=True)
               or getattr(ctx, "project_id", "") or "")


def _carry_room_route(ctx: ToolContext, metadata: Any, pid: str, evt: Dict[str, Any]) -> None:
    """The owner's request moving to its own task in the same room keeps the chat model its room admitted it on
    (``route_choice``); work for another room starts on that room's choice (the promote captures it)."""
    from ouroboros import route_choice
    from ouroboros.project_facts import sanitize_project_id

    if not isinstance(metadata, dict) or str(metadata.get("delegation_role") or "").lower() == "subagent":
        return
    carried = route_choice.room_route_snapshot(metadata.get(route_choice.ROOM_ROUTE_KEY))
    if not carried:
        return
    try:
        same_room = pid == sanitize_project_id(_inherited_project_scope(ctx))
    except (OSError, ValueError):
        return  # an unreadable room proves no sameness: the destination's own choice applies
    if same_room:
        evt[route_choice.ROOM_ROUTE_KEY] = carried


def _attach_client_surface(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Copy the routing turn's per-message client-surface fact onto a
    promote/route/steer event BY VALUE (the origin_message_ref rail's sibling:
    the fact was captured at ingress; producers never re-derive it)."""
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(metadata, dict):
        return
    fact = metadata.get("client_surface")
    if isinstance(fact, dict) and fact:
        evt["client_surface"] = dict(fact)


ISSUER_OWNER_TURN = "owner_turn"
ISSUER_TASK = "task"


def _routing_issuer(ctx: ToolContext) -> Dict[str, Any]:
    """WHO speaks through this routing act -- minted by value where the host knows.

    An OWNER TURN is the direct turn the owner door stamped: ``is_direct_chat`` AND
    ``run_origin``'s ``owner_ingress`` (``origin_message_ref`` / ``origin_suppressed``,
    written only by owner routing). Every other context speaks as a TASK, including
    a pooled root relaying an owner message it just drained. ``last_owner_delivery``
    keys the receipt, never the issuer. The 14.09 incident decided this five times
    from proxies (a routing contract a Swarm root never has, an empty client id read
    as "agent-issued", a room veto keyed on the chat): the host now states it once,
    and the model has no argument to claim otherwise.

    Neither the lane nor a client id makes an owner turn: a consciousness wake-up,
    a Presence event (which carries the provider's event id as its client id) and
    the auto-resume template all run on the direct lane, and nobody typed them; a
    promoted root inherits the owner's stamp as ancestry but is not a direct turn.
    All of them speak as a task.
    """
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    from ouroboros.dialogue_provenance import run_origin

    if bool(getattr(ctx, "is_direct_chat", False)) and run_origin({"metadata": metadata})["owner_ingress"]:
        return {"kind": ISSUER_OWNER_TURN}
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    return {
        "kind": ISSUER_TASK,
        "task_id": task_id,
        "root_task_id": str(metadata.get("root_task_id") or task_id),
    }


def _finish_swarm_handoff(
    ctx: ToolContext,
    evt: Dict[str, Any],
    response: str,
    *,
    status: str,
    reason: str = "",
) -> str:
    """Preserve the first Presence admission receipt for its terminal consumers."""

    metadata = getattr(ctx, "task_metadata", {})
    presence_turn = isinstance(metadata, dict) and bool(metadata.get("presence"))
    if presence_turn and not isinstance(
        getattr(ctx, "_swarm_handoff_attempt", None), dict
    ):
        ctx._swarm_handoff_attempt = {
            "task_id": str(evt.get("task_id") or ""),
            "routing_token": str(evt.get("routing_token") or ""),
            "status": status,
            "reason": reason,
            "response": response,
        }
    return response


def _effective_scope_note(ctx: ToolContext, project_id: str) -> str:
    """Report the separate work's recorded destination, not its requested room."""
    pid = str(project_id or "").strip()
    if not pid:
        return ""
    name = ""
    try:
        from ouroboros.projects_registry import get_project

        name = str((get_project(canonical_data_root(ctx), pid) or {}).get("name") or "").strip()
    except Exception:
        log.debug("promote: effective project name lookup failed", exc_info=True)
    return f" in project '{name}' ({pid})" if name and name != pid else f" in project '{pid}'"


def _requested_scope_label(display_name: str, project_id: str) -> str:
    """What the caller ASKED for, named only where the outcome is unknown."""
    if display_name:
        return f"new project '{display_name}'"
    return f"project '{project_id}'" if project_id else "the main chat"


def _task_card_title_arg(title: str, verb: str) -> tuple[str, str]:
    """Model-created roots need a name that survives admission's lexical cleaner."""
    from ouroboros.project_naming import admission_names

    card_title = str(title or "").strip()[:80]
    if admission_names({"title": card_title}, "")[0]:
        return card_title, ""
    return "", (
        f"⚠️ TOOL_ARG_ERROR ({verb}): title is required: the short name "
        "the owner sees on the new task's card (<=80 chars). Supply title separately "
        "from the executor's work order. Nothing was scheduled."
    )


def _start_task(
    ctx: ToolContext,
    objective: str,
    expected_output: str = "",
    project_id: str = "",
    workspace_root: str = "",
    title: str = "",
    project_name: str = "",
    workspace: str = "",
    source: str = "",
    context_requires_self_body_docs: bool = False,
    reasoning_effort: Any = None,
    context_requires_development: Optional[bool] = None,
) -> str:
    """Start separate new work; the caller keeps its identity and obligations."""
    effort_fields, effort_error, effort_note = _root_effort_arg(reasoning_effort, "start_task")
    if effort_error:
        return effort_error
    goal = str(objective or "").strip()
    if not goal:
        return "⚠️ TOOL_ARG_ERROR (start_task): objective is required"
    card_title, title_error = _task_card_title_arg(title, "start_task")
    if title_error:
        return title_error
    from ouroboros.project_facts import (
        explicit_project_id_ok,
        project_id_from_display_name,
        sanitize_project_id,
    )

    display_name = str(project_name or "").strip()
    pid = ""
    if str(project_id or "").strip():
        if not explicit_project_id_ok(project_id):
            return (
                f"⚠️ TOOL_ARG_ERROR (start_task): project_id {project_id!r} is not "
                "filesystem-clean; use lowercase alphanumeric/_/-/. (<=64 chars)"
            )
        pid = sanitize_project_id(project_id)
    elif display_name:
        # LLM-first "create a NAMED project and work there": derive a filesystem
        # id from the display name. A non-ASCII name (e.g. a Cyrillic "динозавры")
        # falls back to a deterministic hash id so the project is still created —
        # the human-readable name rides project_name on the registry.
        pid = project_id_from_display_name(display_name)
    else:
        # The caller's own room supplies the default, never a sibling's origin.
        pid = sanitize_project_id(_inherited_project_scope(ctx))
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    requested_root = str(workspace_root or "").strip()
    workspace_sentinel = str(workspace or "").strip().lower()
    repo_root_note = ""
    intent_system_repo = False
    if requested_root:
        # Q4=A: naming the Ouroboros repository ITSELF names the documented
        # default (no separate workspace — the ordinary self-modification task),
        # so the EXACT root maps onto the existing "none" sentinel, in Main and
        # in every project room alike. A subfolder, the data drive and every
        # other path pass through unchanged; admission refuses them with the
        # repair hint.
        from ouroboros.tool_access import paths_overlap_casefold
        from ouroboros.workspace_admission import WORKSPACE_NONE

        system_repo = getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None)
        try:
            requested_path = Path(requested_root).expanduser().resolve(strict=False)
            system_repo_path = Path(str(system_repo)).resolve(strict=False) if system_repo else None
            same_root = (
                system_repo_path is not None
                and len(requested_path.parts) == len(system_repo_path.parts)
                and paths_overlap_casefold(requested_path, system_repo_path)
            )
        except (OSError, ValueError, RuntimeError):
            same_root = False
        if same_root:
            requested_root, workspace_sentinel = "", WORKSPACE_NONE
            intent_system_repo = True  # the CHOICE survives the sentinel (#1315)
            repo_root_note = (
                " (workspace_root named the Ouroboros repository itself; started as an "
                "ordinary task over it — no separate workspace)"
            )
    tid = uuid.uuid4().hex[:16]
    routing_token = uuid.uuid4().hex
    disabled_reason = _promotion_pool_disabled_from_snapshot(ctx)
    if disabled_reason:
        response = (
            f"⚠️ PROMOTE_REJECTED: task {tid} was not scheduled "
            f"(worker_pool_unavailable: {disabled_reason}){_cause_words('worker_pool_unavailable')}. "
            "No project/workspace admission side effects were started."
        )
        return _finish_swarm_handoff(
            ctx,
            {"task_id": tid, "routing_token": routing_token},
            response,
            status="rejected",
            reason=f"worker_pool_unavailable:{disabled_reason}",
        )
    evt: Dict[str, Any] = {
        "type": "promote_chat_to_task",
        "task_id": tid,
        "routing_token": routing_token,
        "objective": goal,
        "expected_output": str(expected_output or "").strip(),
        "project_id": pid,
        "project_name": display_name,
        "title": card_title,
        "workspace_root": requested_root,
        # Source admission is intentionally supervisor-side, after the
        # authoritative worker-pool and duplicate-id gates.
        "source": str(source or "").strip(),
        # v6.58.0: "none" opts a project-room task OUT of the room's working_dir
        # default (a folder-less task in a folder-ful project stays possible).
        "workspace": workspace_sentinel,
        "context_requires_self_body_docs": context_requires_self_body_docs,
        # Absent is not false: only an explicit answer overrides the subject-of-work rule (BIBLE P1).
        **({"context_requires_development": bool(context_requires_development)}
           if isinstance(context_requires_development, bool) else {}),
        **effort_fields,
        "chat_id": current_chat_id,
        **({"resource_intent": {"kind": "system_repo"}} if intent_system_repo else {}),
        "client_message_id": str(
            ((getattr(ctx, "task_metadata", {}) or {}).get("client_message_id") or "")
            if isinstance(getattr(ctx, "task_metadata", {}), dict) else ""
        ),
        "attachment_uploads": list(
            ((getattr(ctx, "task_metadata", {}) or {}).get("chat_attachment_uploads") or [])
            if isinstance(getattr(ctx, "task_metadata", {}), dict) else []
        ),
        "ts": utc_now_iso(),
    }
    metadata = getattr(ctx, "task_metadata", {})
    _carry_room_route(ctx, metadata, pid, evt)
    presence_carrier = presence_root_carrier(metadata, task_contract=getattr(ctx, "task_contract", None))
    if presence_carrier:
        contract = dict(getattr(ctx, "task_contract", {}) or {})
        evt.update({**presence_carrier, "task_contract": contract})
        if "capability_ceiling" in contract:
            # Only retained legacy work has an immutable scope ceiling. A new
            # focus keeps ordinary workspace admission and its return carrier.
            evt.update({key: "" for key in ("project_id", "project_name", "workspace_root", "workspace", "source")})
            repo_root_note = ""
    # A promote from a consciousness turn/tree mints a consciousness root: the
    # origin label, ledger category and level ride the event by value; the
    # supervisor stamps them on the new root (worker_promotion) — no presence-style
    # stripping, the wake chooses project/workspace like any Main turn (В9').
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    evt.update(consciousness_origin_metadata({**metadata, "task_id": str(getattr(ctx, "task_id", "") or "")}, started_root=True))
    _attach_origin_from_metadata(ctx, evt)
    _attach_drafted_objective(ctx, evt)
    _attach_client_surface(ctx, evt)
    mode, confirmation = _emit_and_wait_for_routing(ctx, evt)
    confirmation_status = str(confirmation.get("status") or "unconfirmed")
    reason = str(confirmation.get("reason") or "")
    detail = str(confirmation.get("detail") or "")
    disabled_reason = str(confirmation.get("worker_pool_disabled_reason") or "")
    if confirmation_status == "scheduled":
        source_confirmation = f" [{detail}]" if detail else ""
        effective_pid = str(confirmation.get("effective_project_id") or "")
        scope_note = _effective_scope_note(ctx, effective_pid)
        response = (
            f"OK: task {tid}{scope_note} accepted and durably scheduled ({mode}){repo_root_note}."
            f"{source_confirmation} "
            "The task now runs independently, and follow-up chat can steer it. "
            "Use wait_task/get_task_result if its result "
            "is needed in this conversation."
            + (f"\n{effort_note}" if effort_note else "")
        )
        return _finish_swarm_handoff(ctx, evt, response, status="scheduled")
    if confirmation_status in {"rejected", "needs_manual_target", "not_started"}:
        shown_reason = (
            f"{reason}: {disabled_reason}" if disabled_reason else reason
        )
        if detail:
            shown_reason = f"{shown_reason}: {detail}" if shown_reason else detail
        response = (
            f"⚠️ PROMOTE_REJECTED: task {tid} was not scheduled"
            f"{f' ({shown_reason})' if shown_reason else ''}{_cause_words(reason)}. "
            "Do not report this task as created."
        )
        return _finish_swarm_handoff(
            ctx, evt, response, status="rejected", reason=shown_reason or "admission_rejected",
        )
    try:
        root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
        append_jsonl(
            root / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_unconfirmed",
                "task_id": tid,
                "transport_mode": mode,
                "reason": reason or "confirmation_timeout",
                "routing_token": routing_token,
            },
        )
    except Exception:
        log.debug("Failed to record unconfirmed promote", exc_info=True)
    response = (
        f"Accepted, waiting to start: separate task {tid}. Its complete intent is durable; "
        f"admission to {_requested_scope_label(display_name, pid)} is pending ({mode}). "
        "A [ROUTING_RECEIPTS] row reports its recorded outcome. "
        f"Use get_task_result({tid}) to reconcile it before an explicit retry."
    )
    return _finish_swarm_handoff(
        ctx, evt, response, status="unconfirmed", reason=reason or "confirmation_timeout",
    )


@completed_local_read
def _list_projects(ctx: ToolContext, limit: int = 50, closed: Optional[bool] = None,
                   oldest_first: bool = False, start: int = 0) -> str:
    """Enumerate the owner's projects (id, name, recency) so the one mind can
    decide whether a main-chat message belongs to an existing project. The registry
    lives on the CANONICAL data root: a forked execution drive never carries
    ``state/projects.json``, so reading the task's own drive answered "no projects"."""
    if type(start) is not int or start < 0:
        return "⚠️ TOOL_ARG_ERROR (list_projects): start must be a nonnegative integer."
    try:
        from ouroboros.projects_registry import projects_summary
        limit = max(1, min(int(limit or 50), 200))
        rows = projects_summary(canonical_data_root(ctx), limit=limit + 1, start=start,
                                closed=closed, oldest_first=oldest_first)
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if not rows:
        return "No matching projects. move_to_project places this work in a named room."
    lines = []
    for p in rows[:limit]:
        pid = str(p.get("id") or "")
        name = str(p.get("name") or pid)
        last = str(p.get("last_active_at") or p.get("created_at") or "")
        active = " · running" if p.get("has_thread_activity") else ""
        closure = json.dumps(p.get("closed", False), ensure_ascii=False)
        lines.append(f"- {pid} — {name}{active} · last_active_at={last} · closed={closure}")
    if len(rows) > limit:
        args = [f"start={start + limit}", f"limit={limit}", f"oldest_first={oldest_first!r}"]
        if closed is not None:
            args.append(f"closed={closed!r}")
        lines.append("Next: list_projects(" + ", ".join(args) + ")")
    return "Projects (move_to_project places this work in a room):\n" + "\n".join(lines)


def _close_project(ctx: ToolContext, project_id: str, note: str, reopen: bool = False) -> str:
    """Record a project decision and owe its owner one System row in that room."""
    meta = getattr(ctx, "task_metadata", {}) or {}
    lineage = (getattr(ctx, "task_contract", {}) or {}).get("lineage") or {}
    if str(meta.get("delegation_role") or lineage.get("delegation_role") or "") == "subagent":
        return "⚠️ TOOL_ERROR (close_project): subagents inherit the parent's project scope."
    from ouroboros.project_facts import explicit_project_id_ok
    from ouroboros.projects_registry import update_project
    from supervisor import terminal_delivery as outbox

    if not explicit_project_id_ok(project_id) or not isinstance(note, str) or not note.strip() or type(reopen) is not bool:
        return "⚠️ TOOL_ARG_ERROR (close_project): supply a filesystem-clean project_id, a nonempty note and boolean reopen."
    task_id = str(getattr(ctx, "task_id", "") or "")
    if not task_id:
        return "⚠️ TOOL_ERROR (close_project): this work has no task id."
    root, at = canonical_data_root(ctx), utc_now_iso()
    by = f"ouroboros:{task_id}"
    try:
        project, changed = update_project(root, project_id, return_changed=True,
            closed=None if reopen else {"at": at, "by": by, "note": note.strip()}) or (None, False)
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if project is None:
        return "⚠️ PROJECTS_ERROR: project is missing or no longer active."
    state = "open" if reopen else "closed"
    if not changed:
        return f"OK: project {project['id']} is already {state}."
    action = "reopened" if reopen else "closed"
    text = f"Ouroboros ({task_id}) {action} this project. {note.strip()}"
    if not reopen:
        text += " A message from you in this room or a new task will reopen it."
    try:
        delivery = outbox.enqueue_terminal_delivery_outcome(root, {
            "type": "send_message", "chat_id": project["chat_id"],
            "role": "system", "system_type": "project_closure",
            "text": text, "delivery_id": f"project-closure:{uuid.uuid4().hex}",
        }, event_queue=getattr(ctx, "event_queue", None))
    except Exception:
        log.warning("Project closure committed but room notice failed", exc_info=True)
        delivery = outbox.ENQUEUE_UNAVAILABLE
    suffix = ("Room notice queued." if delivery in {outbox.ENQUEUE_QUEUED, outbox.ENQUEUE_ALREADY_DELIVERED}
              else f"Room notice delivery is {delivery}; the project change is saved.")
    return f"OK: project {project['id']} is {state}. {suffix}"


def _cause_words(code: str) -> str:
    """``": <words>"`` for a refusal code the host cause table explains, else "".

    The machine prefix and the code stay exactly where readers parse them; the
    words come from the ONE table the owner's receipts read
    (``project_dialogue.ROUTING_REFUSAL_CAUSES``), so the mind reads the same
    fact the owner does. A code without a row stays bare -- never a near-miss.
    """
    from ouroboros.project_dialogue import ROUTING_REFUSAL_CAUSES

    phrase = ROUTING_REFUSAL_CAUSES.get(str(code or "").strip(), "")
    return f": {phrase}" if phrase else ""


