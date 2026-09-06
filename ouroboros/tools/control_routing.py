"""Routing real work out of a conversation lane into a supervised task.

The model decides WHEN a chat message stops being a conversational answer and
becomes work — a new pooled task, a task inside an existing project, or a
follow-up steered into a task already in flight. These verbs only carry that
decision to the supervisor and report the receipt it returns, including the
rejected and unconfirmed outcomes a caller must not describe as scheduled.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from ouroboros.tool_policy import swarm_router_turn
from ouroboros.tools.control_events import (
    _PROMOTE_CONFIRM_TIMEOUT_SEC,
    _emit_and_wait_for_routing,
    _promotion_pool_disabled_from_snapshot,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


from ouroboros.task_status import load_effective_task_result


_MISSING_PREDECESSOR_SELECTOR = object()


def _predecessor_selector_error(value: Any, tool_name: str) -> str:
    """Require the router to state fresh work or a named continuation."""
    if value is _MISSING_PREDECESSOR_SELECTOR or value is None:
        return (
            f"⚠️ TOOL_ARG_ERROR ({tool_name}): predecessor_task_id is required; "
            "pass an empty string for fresh work or the host-listed result id to continue it"
        )
    return ""


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


def _attach_predecessor_authority_from_metadata(
    ctx: ToolContext, evt: Dict[str, Any], predecessor_task_id: str = "",
) -> str:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    selected_id = str(predecessor_task_id or "").strip()
    if not selected_id:
        return ""
    previous = metadata.get("project_last_task_result")
    manifest = metadata.get("main_routing_manifest")
    candidates = (
        manifest.get("final_results")
        if isinstance(manifest, dict) and isinstance(manifest.get("final_results"), list)
        else [previous] if isinstance(previous, dict) else []
    )
    previous = next((
        row for row in candidates
        if isinstance(row, dict) and str(row.get("task_id") or "") == selected_id
    ), None)
    if not isinstance(previous, dict):
        return (
            "predecessor_task_id is not an addressable result in the host "
            "routing manifest"
        )
    status_root = Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or ctx.drive_root
    ))
    if not load_effective_task_result(status_root, selected_id, materialize_artifacts=False):
        return "the selected predecessor task result is missing or unreadable"
    source = previous.get("authority_source") if isinstance(previous, dict) else None
    from ouroboros.agent_startup_checks import valid_task_result_authority_source

    if valid_task_result_authority_source(source, selected_id):
        evt["predecessor_task_id"] = selected_id
        evt["predecessor_authority_source"] = dict(source)
    else:
        return "the selected predecessor has no readable authority source"
    return ""


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


def _attach_swarm_intent(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Carry host-attested Swarm intent into the admitted managed root."""

    if not swarm_router_turn(ctx):
        return
    metadata = getattr(ctx, "task_metadata", {})
    evt["force_plan"] = True
    evt["force_plan_source"] = str(
        metadata.get("force_plan_source") or "operator"
    ).strip() or "operator"


def _cached_swarm_handoff(ctx: ToolContext) -> str:
    attempt = getattr(ctx, "_swarm_handoff_attempt", None)
    return str(attempt.get("response") or "") if swarm_router_turn(ctx) and isinstance(attempt, dict) else ""


def _finish_swarm_handoff(
    ctx: ToolContext,
    evt: Dict[str, Any],
    response: str,
    *,
    status: str,
    reason: str = "",
) -> str:
    """Latch one immutable admission attempt; repeated calls emit nothing."""

    metadata = getattr(ctx, "task_metadata", {})
    presence_turn = isinstance(metadata, dict) and bool(metadata.get("presence"))
    if (swarm_router_turn(ctx) or presence_turn) and not isinstance(
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


def _promote_chat_to_task(
    ctx: ToolContext,
    objective: str,
    expected_output: str = "",
    project_id: str = "",
    workspace_root: str = "",
    title: str = "",
    project_name: str = "",
    workspace: str = "",
    source: str = "",
    predecessor_task_id: Any = _MISSING_PREDECESSOR_SELECTOR,
) -> str:
    """Route real work out of the conversation lane into a supervised pooled task.

    Option B of the multi-project chat plane (v6.32.0): the conversation stays
    in the fast in-process lane; ANY substantial work spawns a first-class
    pooled task with a live card. The decision is the model's own structural
    tool call (BIBLE P5 — no keyword routing). Follow-up owner messages reach
    the running task through its owner-mailbox.

    ``title`` is a short human name the model coins for the card AT CREATION
    (no extra request, owner P1) — reused as the project name if this task is
    later turned into a project. ``project_name`` makes this an LLM-first
    "create a named project and work there" call: the project is created NOW
    with that display name and the task runs inside it (v6.33.0).
    """
    selector_error = _predecessor_selector_error(predecessor_task_id, "promote_chat_to_task")
    if selector_error:
        return selector_error
    goal = str(objective or "").strip()
    if not goal:
        return "⚠️ TOOL_ARG_ERROR (promote_chat_to_task): objective is required"
    cached = _cached_swarm_handoff(ctx)
    if cached:
        return cached
    from ouroboros.project_facts import (
        explicit_project_id_ok,
        project_id_from_display_name,
        sanitize_project_id,
    )

    scope_override_note = ""
    if swarm_router_turn(ctx):
        # The model chooses admission; the host-owned room chooses scope — but
        # room scope wins only on a GENUINE conflict (room already bound to a
        # project). In a projectless room an explicitly passed project_name OR
        # project_id is INHERITED (Q9-A): silently clearing them made the
        # saga's first root run projectless, so its work landed in an
        # off-registry tree that no later task could see.
        room_pid = str(getattr(ctx, "project_id", "") or "")
        if room_pid:
            explicit = str(project_name or "").strip() or str(project_id or "").strip()
            explicit_pid = (
                project_id_from_display_name(project_name)
                if str(project_name or "").strip()
                else sanitize_project_id(project_id or "")
            )
            if explicit and explicit_pid != room_pid:
                # An explicit owner input lost to the room binding — disclose
                # it in the response, never drop silently (the silent drop was
                # the saga defect).
                scope_override_note = (
                    f" Explicit project {explicit!r} was ignored: this room is "
                    f"bound to project {room_pid!r}."
                )
            project_id = room_pid
            project_name = ""
        workspace_root = workspace = source = ""

    display_name = str(project_name or "").strip()
    pid = ""
    if str(project_id or "").strip():
        if not explicit_project_id_ok(project_id):
            return (
                f"⚠️ TOOL_ARG_ERROR (promote_chat_to_task): project_id {project_id!r} is not "
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
        # No explicit arg: inherit the CURRENT project scope so a project-chat
        # task that promotes follow-up work stays in its own project (the model
        # still chose to promote — scope is contextual, never a keyword gate).
        pid = sanitize_project_id(getattr(ctx, "project_id", "") or "")
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    tid = uuid.uuid4().hex[:16]
    routing_token = uuid.uuid4().hex
    disabled_reason = _promotion_pool_disabled_from_snapshot(ctx)
    if disabled_reason:
        response = (
            f"PROMOTE_REJECTED: task {tid} was not scheduled "
            f"(worker_pool_unavailable: {disabled_reason}). No project/workspace "
            "admission side effects were started."
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
        "title": str(title or "").strip()[:80],
        "workspace_root": str(workspace_root or "").strip(),
        # Source admission is intentionally supervisor-side, after the
        # authoritative worker-pool and duplicate-id gates.
        "source": str(source or "").strip(),
        # v6.58.0: "none" opts a project-room task OUT of the room's working_dir
        # default (a folder-less task in a folder-ful project stays possible).
        "workspace": str(workspace or "").strip().lower(),
        "chat_id": current_chat_id,
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
    presence = metadata.get("presence") if isinstance(metadata, dict) else None
    if isinstance(presence, dict) and presence:
        # A public conversation may promote long work, but it cannot choose a
        # new Project/workspace/source authority. The immutable positive ceiling
        # and exact return destination follow the promoted root by value.
        evt.update({
            "project_id": "",
            "project_name": "",
            "workspace_root": "",
            "workspace": "",
            "source": "",
            "presence": dict(presence),
            "task_contract": dict(getattr(ctx, "task_contract", {}) or {}),
        })
    _attach_origin_from_metadata(ctx, evt)
    predecessor_error = _attach_predecessor_authority_from_metadata(
        ctx, evt, predecessor_task_id,
    )
    if predecessor_error:
        return (
            "⚠️ AUTHORITY_SOURCE_UNAVAILABLE (promote_chat_to_task): "
            + predecessor_error
        )
    _attach_swarm_intent(ctx, evt)
    _attach_client_surface(ctx, evt)
    mode, confirmation = _emit_and_wait_for_routing(ctx, evt)
    if display_name:
        scope_note = f" in new project '{display_name}'"
    elif pid:
        scope_note = f" in project '{pid}'"
    else:
        scope_note = ""
    confirmation_status = str(confirmation.get("status") or "unconfirmed")
    reason = str(confirmation.get("reason") or "")
    detail = str(confirmation.get("detail") or "")
    disabled_reason = str(confirmation.get("worker_pool_disabled_reason") or "")
    if confirmation_status == "scheduled":
        source_confirmation = f" [{detail}]" if detail else ""
        response = (
            f"OK: task {tid}{scope_note} accepted and durably scheduled ({mode}).{source_confirmation} "
            "The task now runs independently, and follow-up chat can steer it. "
            "Use wait_task/get_task_result if its result "
            "is needed in this conversation." + scope_override_note
        )
        return _finish_swarm_handoff(ctx, evt, response, status="scheduled")
    if confirmation_status in {"rejected", "needs_manual_target"}:
        shown_reason = (
            f"{reason}: {disabled_reason}" if disabled_reason else reason
        )
        if detail:
            shown_reason = f"{shown_reason}: {detail}" if shown_reason else detail
        response = (
            f"PROMOTE_REJECTED: task {tid} was not scheduled"
            f"{f' ({shown_reason})' if shown_reason else ''}. "
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
    confirmation_window = (
        f"within {int(_PROMOTE_CONFIRM_TIMEOUT_SEC)} seconds"
        if mode == "live"
        else f"because the event transport returned {mode}"
    )
    response = (
        f"PROMOTE_UNCONFIRMED: task {tid} admission was not confirmed {confirmation_window}. "
        "Do not report this task as "
        "created and do not retry automatically; keep this task id for reconciliation."
    )
    return _finish_swarm_handoff(
        ctx, evt, response, status="unconfirmed", reason=reason or "confirmation_timeout",
    )


def _list_projects(ctx: ToolContext, limit: int = 50) -> str:
    """Enumerate the owner's projects (id, name, recency) so the one mind can
    decide whether a main-chat message belongs to an existing project."""
    try:
        from ouroboros.projects_registry import projects_summary
        rows = projects_summary(Path(ctx.drive_root), limit=max(1, min(int(limit or 50), 200)))
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if not rows:
        return "No projects yet. Create one by promoting work with a fresh project_id, or just answer/spawn a task."
    lines = []
    for p in rows:
        pid = str(p.get("id") or "")
        name = str(p.get("name") or pid)
        last = str(p.get("last_active_at") or p.get("created_at") or "")
        active = " · running" if p.get("has_thread_activity") else ""
        lines.append(f"- {pid} — {name}{active}{(' · last ' + last) if last else ''}")
    return "Projects (route a related main-chat message with route_to_project):\n" + "\n".join(lines)


def _route_to_project(
    ctx: ToolContext, project_id: str = "", message: str = "", reason: str = "",
    predecessor_task_id: Any = _MISSING_PREDECESSOR_SELECTOR,
    candidates: Any = None,
) -> str:
    """Route a main-chat message to an EXISTING project so the work continues in
    that project's context (its memory/journal/thread), keeping the main chat free.

    LLM-first: the model decides WHEN to route (its judgment is the gate, never a
    keyword rule); this verb just delivers the decision and returns a visible
    receipt. The receipt is host metadata on the owner message; any non-empty
    final decision-turn explanation remains a separate conversational reply.
    """
    selector_error = _predecessor_selector_error(predecessor_task_id, "route_to_project")
    if selector_error:
        return selector_error
    from ouroboros.project_facts import explicit_project_id_ok, sanitize_project_id
    from ouroboros.projects_registry import get_project

    msg = str(message or "")
    if not msg.strip():
        return "⚠️ TOOL_ARG_ERROR (route_to_project): message is required"
    cached = _cached_swarm_handoff(ctx)
    if cached:
        return cached
    if swarm_router_turn(ctx) and str(getattr(ctx, "project_id", "") or "").strip():
        return (
            "⚠️ SWARM_PROJECT_SCOPE_OWNED: this Project-room Swarm must create its new "
            "root with promote_chat_to_task in the current Project."
        )
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    routing_contract = (
        metadata.get("routing_contract")
        if isinstance(metadata.get("routing_contract"), dict)
        else {}
    )
    client_message_id = str(metadata.get("client_message_id") or "").strip()
    predecessor_event: Dict[str, Any] = {}
    predecessor_error = _attach_predecessor_authority_from_metadata(
        ctx, predecessor_event, predecessor_task_id,
    )
    if predecessor_error:
        return "⚠️ AUTHORITY_SOURCE_UNAVAILABLE (route_to_project): " + predecessor_error
    requested_pid = str(project_id or "").strip()
    pid = sanitize_project_id(requested_pid) if requested_pid and explicit_project_id_ok(requested_pid) else ""
    proj = get_project(Path(ctx.drive_root), pid) if pid else None
    if not proj:
        # The decision actor cannot manufacture a UI payload by returning prose.
        # An empty, malformed, or stale target becomes the typed manual-target
        # control event, carrying only the host-built options from this turn.
        options = [
            dict(row) for row in list(routing_contract.get("manual_options") or [])[:100]
            if isinstance(row, dict)
        ]
        # Owner decision 2=B: the model may NARROW the picker by naming its
        # plausible candidates — a host-validated reorder, never new options.
        # Named ids that match host options move to the front; unknown ids are
        # ignored (host truth wins), and every host option stays clickable.
        candidate_ids = list(dict.fromkeys(
            str(row).strip() for row in (candidates if isinstance(candidates, list) else [])
            if str(row).strip()
        ))
        if candidate_ids:
            def _option_id(row: Dict[str, Any]) -> str:
                return str(row.get("task_id") or row.get("project_id") or "")

            ranked = [
                row for cid in candidate_ids for row in options if _option_id(row) == cid
            ]
            ranked_ids = {id(row) for row in ranked}
            options = ranked + [row for row in options if id(row) not in ranked_ids]
        failure = (
            "target_unspecified" if not requested_pid
            else "invalid_project_id" if not pid
            else "target_not_found"
        )
        routing_token = uuid.uuid4().hex
        manual_event: Dict[str, Any] = {
            "type": "routing_manual_target",
            "routing_token": routing_token,
            "chat_id": current_chat_id,
            "client_message_id": client_message_id,
            "requested_target": pid or requested_pid[:200],
            "reason": str(reason or "").strip() or failure,
            "options": options,
            # The picker click dispatches AFTER this turn's metadata is gone,
            # so the refusal annotation is the durable carrier of the original
            # message's staged-attachment specs (#198).
            "attachment_uploads": list(
                ((getattr(ctx, "task_metadata", {}) or {}).get("chat_attachment_uploads") or [])
            ),
            "ts": utc_now_iso(),
        }
        manual_event.update(predecessor_event)
        mode, receipt = _emit_and_wait_for_routing(ctx, manual_event)
        if str(receipt.get("status") or "") == "needs_manual_target":
            durable_options = (
                receipt.get("options") if isinstance(receipt.get("options"), list) else options
            )
            options_text = json.dumps(durable_options, ensure_ascii=False, default=str)
            return (
                f"⚠️ NEEDS_MANUAL_TARGET ({failure}, {mode}): no route was dispatched. "
                f"Host-validated options: {options_text}"
            )
        return (
            f"⚠️ ROUTING_UNCONFIRMED ({failure}, {mode}): no route was dispatched and "
            "delivery of the manual target options was not confirmed."
        )
    tid = uuid.uuid4().hex[:16]
    routing_token = uuid.uuid4().hex
    objective = msg if not str(reason or "").strip() else f"{msg}\n\n(routing reason: {str(reason).strip()})"
    evt: Dict[str, Any] = {
        "type": "promote_chat_to_task",
        "task_id": tid,
        "routing_token": routing_token,
        "objective": objective,
        "project_id": pid,
        "chat_id": current_chat_id,
        "routed_from_main": True,
        "client_message_id": client_message_id,
        "attachment_uploads": list(
            ((getattr(ctx, "task_metadata", {}) or {}).get("chat_attachment_uploads") or [])
            if isinstance(getattr(ctx, "task_metadata", {}), dict) else []
        ),
        "ts": utc_now_iso(),
    }
    _attach_origin_from_metadata(ctx, evt)
    evt.update(predecessor_event)
    _attach_swarm_intent(ctx, evt)
    _attach_client_surface(ctx, evt)
    mode, receipt = _emit_and_wait_for_routing(ctx, evt)
    name = str(proj.get("name") or pid)
    status = str(receipt.get("status") or "unconfirmed")
    if status == "scheduled":
        response = (
            f"✉️ Routed to project '{name}' ({pid}) as task {tid}; admission is durably "
            f"scheduled ({mode}). I'll continue there; this chat stays free for you."
        )
        return _finish_swarm_handoff(ctx, evt, response, status="scheduled")
    reason_text = str(receipt.get("reason") or "confirmation_timeout")
    detail = str(receipt.get("detail") or "")
    if status in {"rejected", "needs_manual_target"}:
        response = (
            f"⚠️ ROUTE_REJECTED: task {tid} was not routed to project '{name}' "
            f"({reason_text}{(': ' + detail) if detail else ''})."
        )
        return _finish_swarm_handoff(
            ctx, evt, response, status="rejected", reason=reason_text,
        )
    response = (
        f"⚠️ ROUTE_UNCONFIRMED: task {tid} routing to project '{name}' was not durably "
        "confirmed. Do not report it as routed and do not retry automatically."
    )
    return _finish_swarm_handoff(
        ctx, evt, response, status="unconfirmed", reason=reason_text,
    )


def _steer_task(ctx: ToolContext, task_id: str, message: str) -> str:
    """Deliver a follow-up to a host-listed RUNNING/PENDING owner root.

    Project rooms are limited to ``current_chat.addressable_root_tasks``; Main
    may also choose a Project-bound root from ``main_routing_manifest.root_tasks``.

    When the chat is busy, a new message runs as a short-lived decision turn that
    sees the running tasks of the current chat as structural context and picks the
    one to steer. This verb just transports the message to that task's owner-mailbox
    (the running task drains it at its next safe checkpoint). LLM-first (BIBLE P5):
    the code never decides which task a message belongs to — it only validates the
    transport (task exists, same chat, idempotent delivery) and the supervisor
    performs the mailbox write on the task's active drive. When unsure which task
    (or none) fits, spawn a fresh task with ``promote_chat_to_task`` instead.
    """
    if swarm_router_turn(ctx):
        return (
            "⚠️ SWARM_NEW_ROOT_REQUIRED: explicit Swarm cannot steer an existing task; "
            "use promote_chat_to_task or, from Main, route_to_project."
        )
    target = str(task_id or "").strip()
    msg = str(message or "").strip()
    if not target:
        return (
            "⚠️ TOOL_ARG_ERROR (steer_task): task_id is required — pick one from "
            "current_chat.running_tasks (or promote_chat_to_task to start new work)."
        )
    if not msg.strip():
        return "⚠️ TOOL_ARG_ERROR (steer_task): message is required."
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    _md = getattr(ctx, "task_metadata", None)
    # The model chooses the target, but the host transports the exact owner
    # bytes captured at ingress.  A model-authored paraphrase must not replace
    # the owner's steering text.  Non-owner/internal calls have no origin text
    # and retain the explicit tool argument.
    if isinstance(_md, dict) and isinstance(_md.get("origin_message_text"), str):
        exact_owner_text = str(_md.get("origin_message_text") or "")
        if exact_owner_text.strip():
            msg = exact_owner_text
    client_message_id = str((_md.get("client_message_id") if isinstance(_md, dict) else "") or "").strip()
    routing_contract = (
        _md.get("routing_contract")
        if isinstance(_md, dict) and isinstance(_md.get("routing_contract"), dict)
        else {}
    )
    evt: Dict[str, Any] = {
        "type": "steer_task",
        "routing_token": uuid.uuid4().hex,
        "target_task_id": target,
        "message": msg,
        "chat_id": current_chat_id,
        "client_message_id": client_message_id,
        # Main sees the global root manifest, including Project-bound roots.  The
        # flag is derived from host metadata (not a model argument), allowing the
        # supervisor to validate that exact documented addressability.
        "allow_global_root": routing_contract.get("source_lane") == "main",
        "attachment_uploads": list(_md.get("chat_attachment_uploads") or [])
        if isinstance(_md, dict) else [],
        "ts": utc_now_iso(),
    }
    _attach_client_surface(ctx, evt)
    mode, receipt = _emit_and_wait_for_routing(ctx, evt)
    status = str(receipt.get("status") or "unconfirmed")
    if status == "delivered":
        confirmation = (
            f"✉️ Steering task {target}: mailbox delivery is durably confirmed ({mode}). "
            "The task receives it at its next checkpoint."
        )
        detail = str(receipt.get("detail") or "")
        if detail:
            confirmation += f"\n\n[ATTACHMENTS]\n{detail}\n[END_ATTACHMENTS]"
        return confirmation
    if status in {"rejected", "needs_manual_target"}:
        return (
            f"⚠️ STEER_REJECTED: task {target} was not steered "
            f"({str(receipt.get('reason') or 'target_not_steerable')})."
        )
    return (
        f"⚠️ STEER_UNCONFIRMED: mailbox delivery to task {target} was not durably confirmed "
        f"({mode}). Do not report the message as delivered."
    )
