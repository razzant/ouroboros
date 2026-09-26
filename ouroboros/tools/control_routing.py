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

from ouroboros.dialogue_provenance import presence_root_carrier
from ouroboros.tools.control_events import (
    _PROMOTE_CONFIRM_TIMEOUT_SEC,
    _emit_and_wait_for_routing,
    _promotion_pool_disabled_from_snapshot,
)
from ouroboros.tool_access_paths import canonical_data_root
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
            "pass an empty string for fresh work or the id of a settled result to continue it"
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


def _attach_drafted_objective(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Keep the routed objective's author separate from the ingress owner corpus."""
    evt["objective_author"] = {"kind": "task", "task_id": str(getattr(ctx, "task_id", "") or "")}
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
    evt["owner_corpus"] = owner_rows


def _durable_project_of_request(ctx: ToolContext) -> str:
    """The project this request's work ALREADY has, durably: the promoting task's
    own binding — the one truth about a task's project, which a "Turn into
    project" conversion writes without ever reaching the live worker's
    ``ctx.project_id`` — else the project the OWNER MESSAGE it came from has.

    Both reads fail OPEN exactly like ``project_facts._bound_project_id`` (one
    DEBUG line, then ""): this runs on a routing decision the owner is waiting
    for, and an unreadable store must not stop the work."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.projects_registry import project_id_for_origin, project_id_for_task

        return str(
            project_id_for_task(DATA_DIR, str(getattr(ctx, "task_id", "") or ""))
            or project_id_for_origin(DATA_DIR, metadata.get("origin_message_ref"))
            or ""
        )
    except Exception:
        log.debug("promote: durable project scope lookup failed", exc_info=True)
        return ""


def _inherited_project_scope(ctx: ToolContext) -> str:
    """The project a promote with NO explicit target should land in: the durable
    project of the request (own binding, then the owner message's), so a root
    promoted out of an already-converted message joins it instead of appearing
    in Main as a second convertible unit; else the in-memory scope copy,
    unchanged behaviour. Explicit ``project_id``/``project_name`` never reach
    here — they stay the model's ceiling (BIBLE P13)."""
    return _durable_project_of_request(ctx) or str(getattr(ctx, "project_id", "") or "")


def _host_listed_predecessors(metadata: Dict[str, Any]) -> list:
    """Every result THIS turn's host facts put in front of the model: the Main
    lane's manifest, the room's own hint list, and the room's pointer row."""
    rows: list = []
    for key in ("main_routing_manifest", "project_routing_manifest"):
        manifest = metadata.get(key)
        if isinstance(manifest, dict) and isinstance(manifest.get("final_results"), list):
            rows.extend(row for row in manifest["final_results"] if isinstance(row, dict))
    previous = metadata.get("project_last_task_result")
    if isinstance(previous, dict) and previous:
        rows.append(previous)
    return rows


def _predecessor_door_refusal(result: Dict[str, Any]) -> str:
    """Why this readable result may NOT be continued, or ``""``.

    A PREDICATE on the result itself - settled, and a result rather than a pending
    admission - never on where the caller sits, where the work lands or whether it
    is a root's or a helper's: the pointer is rebuilt from the task id alone, the
    successor inherits DATA (ceiling, origin and contract come from the caller and
    admission), and a fresh root in the predecessor's project was always reachable.
    The host list stays a hint; a foreign landing or a helper is disclosed, never refused.
    """
    from ouroboros.routing_wait import is_emitted_admission_stub
    from ouroboros.task_status import SETTLED_STATUSES

    status = str(result.get("status") or "")
    if is_emitted_admission_stub(result):
        return ("the selected predecessor is a promote whose admission is still pending, not a "
                "result; read get_task_result on it, and name a settled result instead")
    if status not in SETTLED_STATUSES:
        return (
            f"the selected predecessor is still live (status {status or 'unknown'}); "
            "steer_task continues a live root, and promoting it would start a second one"
        )
    return ""


def _predecessor_notes(predecessor_id: str, facts: Dict[str, Any], landed: str) -> str:
    """What the receipt says about the predecessor once, like the second-project note: a
    helper's result names its root (or parent), a foreign landing names both projects."""
    if not predecessor_id:
        return ""
    home, root, parent = (str(facts.get(k) or "") for k in ("project_id", "root_task_id", "parent_task_id"))
    lineage = f"its root is {root}" if root else (f"its parent is {parent}" if parent else "its root is unknown")
    notes = f" Note: predecessor {predecessor_id} is a delegated helper's result; {lineage}." if facts.get("helper") else ""
    if home != str(landed or ""):
        where = f"project '{home}'" if home else "the main chat"
        here = f"project '{landed}'" if landed else "the main chat"
        notes += f" Note: predecessor {predecessor_id} belongs to {where}; this continuation runs in {here} (your choice)."
    return notes


def _attach_predecessor_authority_from_metadata(
    ctx: ToolContext, evt: Dict[str, Any], predecessor_task_id: str = "",
) -> str:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    selected_id = str(predecessor_task_id or "").strip()
    if not selected_id:
        return ""
    listed = next((
        row for row in _host_listed_predecessors(metadata)
        if str(row.get("task_id") or "") == selected_id
    ), None)
    result = load_effective_task_result(canonical_data_root(ctx), selected_id, materialize_artifacts=False)
    if not isinstance(result, dict) or not result:
        return "the selected predecessor task result is missing or unreadable"
    refusal = _predecessor_door_refusal(result)
    if refusal:
        return refusal
    if listed is not None:
        # A row the host showed carries its own host-issued pointer, and only that
        # one is accepted for it: rebuilding a source for a shown row would make a
        # tampered manifest row indistinguishable from a host-built one.
        source = listed.get("authority_source")
    else:
        from ouroboros.server_routing_context import _task_result_ground_truth

        source = _task_result_ground_truth(result).get("authority_source")
    from ouroboros.agent_startup_checks import valid_task_result_authority_source

    if valid_task_result_authority_source(source, selected_id):
        from ouroboros.server_routing_context import _is_child_result

        evt["predecessor_task_id"] = selected_id
        evt["predecessor_authority_source"] = dict(source)
        # Render-only facts for the receipt's notes; the caller pops them before
        # emission, so the event carries nothing the supervisor never reads.
        evt["predecessor_facts"] = {
            "project_id": str(result.get("project_id") or ""), "helper": _is_child_result(result),
            "root_task_id": str(result.get("root_task_id") or ""),
            "parent_task_id": str(result.get("parent_task_id") or ""),
        }
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
    """Where an admitted promote ACTUALLY landed, read from the admission receipt.

    The requested ``project_name``/``project_id`` is the model's ask, not the
    destination: an implicit promote's scope is re-resolved by the admission
    handler under the origin claim lock, so a sibling card converted in the
    emit → admission window moves the root into a project the tool never named.
    Reporting the request as the outcome told the owner the work was in a room it
    was not in. Empty project id means Main, which is also the truth when a
    project scope was requested and admission granted none."""
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

    The conversation stays in the fast in-process lane and keeps its own tools;
    the model promotes when an independent task is useful (SYSTEM.md, Decision
    Loop), and the promoted work runs as a first-class pooled task with a live
    card. The decision is the model's own structural tool call (BIBLE P5 — no
    keyword routing). Follow-up owner messages reach the running task through
    its owner-mailbox.

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
        # The durable binding of this task, then of the owner message it came
        # from, outrank the in-memory copy; see _inherited_project_scope.
        pid = sanitize_project_id(_inherited_project_scope(ctx))
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    requested_root = str(workspace_root or "").strip()
    workspace_sentinel = str(workspace or "").strip().lower()
    repo_root_note = ""
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
        "workspace_root": requested_root,
        # Source admission is intentionally supervisor-side, after the
        # authoritative worker-pool and duplicate-id gates.
        "source": str(source or "").strip(),
        # v6.58.0: "none" opts a project-room task OUT of the room's working_dir
        # default (a folder-less task in a folder-ful project stays possible).
        "workspace": workspace_sentinel,
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
    presence_carrier = presence_root_carrier(metadata, task_contract=getattr(ctx, "task_contract", None))
    if presence_carrier:
        # A public conversation cannot choose a new Project/workspace/source authority; the immutable
        # ceiling and return destination (a descendant's root: its binding only) follow it by value.
        evt.update({
            "project_id": "",
            "project_name": "",
            "workspace_root": "",
            "workspace": "",
            "source": "",
            **presence_carrier,
            "task_contract": dict(getattr(ctx, "task_contract", {}) or {}),
        })
        repo_root_note = ""  # Presence runs in its admitted folder, never over the repo
    # A promote from a consciousness turn/tree mints a consciousness root: the
    # origin label, ledger category and level ride the event by value; the
    # supervisor stamps them on the new root (worker_promotion) — no presence-style
    # stripping, the wake chooses project/workspace like any Main turn (В9').
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    evt.update(consciousness_origin_metadata(metadata))
    _attach_origin_from_metadata(ctx, evt)
    _attach_drafted_objective(ctx, evt)
    predecessor_error = _attach_predecessor_authority_from_metadata(
        ctx, evt, predecessor_task_id,
    )
    if predecessor_error:
        return (
            "⚠️ AUTHORITY_SOURCE_UNAVAILABLE (promote_chat_to_task): "
            + predecessor_error
        )
    predecessor_facts = dict(evt.pop("predecessor_facts", None) or {})
    _attach_client_surface(ctx, evt)
    _attach_unmet_obligation(ctx, evt)
    already_bound = _durable_project_of_request(ctx)
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
            + _second_project_note(ctx, already_bound, effective_pid)
            + _predecessor_notes(str(evt.get("predecessor_task_id") or ""), predecessor_facts, effective_pid)
            + _obligation_moved_note(ctx, tid, confirmation.get("force_plan_transfer"))
        )
        return _finish_swarm_handoff(ctx, evt, response, status="scheduled")
    if confirmation_status in {"rejected", "needs_manual_target"}:
        shown_reason = (
            f"{reason}: {disabled_reason}" if disabled_reason else reason
        )
        if detail:
            shown_reason = f"{shown_reason}: {detail}" if shown_reason else detail
        response = (
            f"⚠️ PROMOTE_REJECTED: task {tid} was not scheduled"
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
        f"⚠️ PROMOTE_UNCONFIRMED: task {tid} admission was not confirmed {confirmation_window}; "
        f"the requested destination was {_requested_scope_label(display_name, pid)} and the "
        "effective one is unknown until the admission is reconciled. Do not report this task as "
        f"created and do not retry automatically: call get_task_result({tid}) to read the durable "
        "outcome before promoting the same work again."
    )
    return _finish_swarm_handoff(
        ctx, evt, response, status="unconfirmed", reason=reason or "confirmation_timeout",
    )


def _attach_unmet_obligation(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Owner 3=A: a task whose Swarm planning obligation is still UNMET carries it
    onto the root it promotes (the existing admission seam stamps the new root
    and releases the promoter in the same transaction). An owner turn has no
    obligation of its own; a met or unreadable one stays where it is."""
    if _routing_issuer(ctx)["kind"] != ISSUER_TASK:
        return
    from ouroboros.owner_hurry import unmet_force_plan_obligation

    obligation = unmet_force_plan_obligation(ctx)
    if obligation.get("unmet"):
        evt.update({
            "force_plan": True,
            "force_plan_source": str(obligation.get("source") or "operator"),
            "force_plan_transferred_from": str(getattr(ctx, "task_id", "") or ""),
        })


def _obligation_moved_note(ctx: ToolContext, new_task_id: str, transfer: Any) -> str:
    """Tell the promoter its obligation moved, and release the worker's own copy."""
    if not isinstance(transfer, dict) or str(transfer.get("from") or "") != str(getattr(ctx, "task_id", "") or ""):
        return ""
    from ouroboros.owner_hurry import release_force_plan_obligation

    release_force_plan_obligation(ctx, new_task_id)
    return (
        f" Your planning obligation (force_plan) moved to task {new_task_id}: it now owes the "
        "plan review, and this task no longer does. If you meant to move THIS task into a "
        "project, use ensure_project_scope; any work you keep doing here yourself is unplanned."
    )


def _second_project_note(ctx: ToolContext, already_bound: str, effective_pid: str) -> str:
    """Owner 5A: promote stays a free choice, so when the request's work already had
    a Project and this promote landed in another one, say so instead of hiding it."""
    if not already_bound or not effective_pid or effective_pid == already_bound:
        return ""
    return (
        f" Note: the request that started this work already has project '{already_bound}'; "
        f"a second project '{effective_pid}' now holds this promote (your choice)."
    )


def _list_projects(ctx: ToolContext, limit: int = 50) -> str:
    """Enumerate the owner's projects (id, name, recency) so the one mind can
    decide whether a main-chat message belongs to an existing project. The registry
    lives on the CANONICAL data root: a forked execution drive never carries
    ``state/projects.json``, so reading the task's own drive answered "no projects"."""
    try:
        from ouroboros.projects_registry import projects_summary
        rows = projects_summary(canonical_data_root(ctx), limit=max(1, min(int(limit or 50), 200)))
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
    predecessor_facts = dict(predecessor_event.pop("predecessor_facts", None) or {})
    requested_pid = str(project_id or "").strip()
    pid = sanitize_project_id(requested_pid) if requested_pid and explicit_project_id_ok(requested_pid) else ""
    proj = get_project(canonical_data_root(ctx), pid) if pid else None
    failure = (
        "target_unspecified" if not requested_pid
        else "invalid_project_id" if not pid
        else "target_not_found"
    )
    if not proj and _routing_issuer(ctx)["kind"] == ISSUER_TASK:
        # A picker is an owner surface (7=A): a task speaking for itself gets
        # the typed refusal in its own result, and no ack travels to a chat under
        # an empty message id. `list_projects` names the ids it may route to.
        return (
            f"⚠️ ROUTE_REJECTED ({failure}): no route was dispatched. A task-authored route "
            "needs an existing project id (see list_projects); the manual-target picker is "
            "an owner surface and is not offered to a task."
        )
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
        routing_token = uuid.uuid4().hex
        manual_event: Dict[str, Any] = {
            "type": "routing_manual_target",
            "routing_token": routing_token,
            "chat_id": current_chat_id,
            "client_message_id": client_message_id,
            "requested_target": pid or requested_pid[:200],
            # The typed code is the receipt's `reason` (the host cause table
            # reads it); the model's own words ride `detail` beside it instead
            # of replacing the code with untyped prose.
            "reason": failure,
            "detail": str(reason or "").strip()[:1000],
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
    # A route mints a root exactly like a promote, so a consciousness turn's origin,
    # ledger category and level ride THIS event too. The single admission door reads
    # them off the event (supervisor/worker_promotion.promote_chat_to_task) and stamps
    # the root; without them a wake's routed root landed with empty metadata, no
    # disabled_tools in its contract and the ordinary `task` ledger category.
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    evt.update(consciousness_origin_metadata(metadata))
    _attach_origin_from_metadata(ctx, evt)
    _attach_drafted_objective(ctx, evt)
    evt.update(predecessor_event)
    _attach_client_surface(ctx, evt)
    # Owner 3=A holds on this verb too: a route starts a NEW root exactly like a
    # promote, so an unmet Swarm planning obligation follows the work into the
    # project instead of staying with a sender that keeps none of it.
    _attach_unmet_obligation(ctx, evt)
    mode, receipt = _emit_and_wait_for_routing(ctx, evt)
    name = str(proj.get("name") or pid)
    status = str(receipt.get("status") or "unconfirmed")
    if status == "scheduled":
        response = (
            f"✉️ Routed to project '{name}' ({pid}) as task {tid}; admission is durably "
            f"scheduled ({mode}). I'll continue there; this chat stays free for you."
            + _predecessor_notes(str(evt.get("predecessor_task_id") or ""), predecessor_facts,
                                 str(receipt.get("effective_project_id") or pid))
            + _obligation_moved_note(ctx, tid, receipt.get("force_plan_transfer"))
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


def _origin_already_routed(ctx: ToolContext, client_message_id: str) -> bool:
    """Whether this turn has ALREADY carried its origin message somewhere.

    The FIRST routing act of a turn is the one that relays the owner's exact
    ingress bytes.  Once that act landed, the turn's later words are its own —
    a pacing note, a hand-off, an answer to something it learned since — and
    replaying the origin over them delivers a stale message nobody wrote
    (#896).  Read from the durable annotation receipts: a LANDED
    promote/route/steer receipt on the origin message.  A refused or
    unconfirmed act carried nothing, so the owner's message is still unrouted
    and the next act still relays it.
    """
    message_id = str(client_message_id or "").strip()
    if not message_id:
        return False
    try:
        from ouroboros.project_dialogue import latest_chat_annotations

        root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
        row = latest_chat_annotations(root).get(message_id) or {}
    except Exception:
        log.debug("Origin routing-receipt lookup failed", exc_info=True)
        return False
    return (
        str(row.get("action") or "") in {"promote_chat_to_task", "route_to_project", "steer_task"}
        and str(row.get("status") or "") in {"scheduled", "delivered"}
    )


def _steer_task(ctx: ToolContext, task_id: str, message: str) -> str:
    """Deliver a message to a host-listed RUNNING/PENDING independent root.

    Who speaks decides how the message travels (``_routing_issuer``). An OWNER
    TURN steers with owner text: the message lands as ``[Message from my
    human]``, enters the owner corpus and supersedes a reviewed answer; the room
    veto and the owner acknowledgement apply. A TASK speaking for itself never
    travels as owner text: its words land as ``[Message from independent task
    <id>]`` -- context the receiving model judges -- through the same writer
    ``forward_to_worker`` uses, and any host-listed active independent root is
    addressable (owner 6C), the hidden partition included.

    A conversation sees the running tasks as structural context and chooses
    which one to steer. LLM-first (BIBLE P5): the code never decides which task
    a message belongs to — it only validates the transport (task exists,
    idempotent delivery) and the supervisor performs the mailbox write on the
    task's active drive. When unsure which task (or none) fits, spawn a fresh
    task with ``promote_chat_to_task`` instead.

    On an owner turn's FIRST routing act, while it is still acting on the
    message that started it, the host delivers that owner message's exact
    ingress bytes instead of any paraphrase. Afterwards — once the turn has
    already routed that message, or a later owner message has actually reached
    it — the message given here is delivered verbatim, so relay the owner's
    words rather than a summary of them. Delivery stays confirmable either way:
    a steer that belongs to no owner message earns its receipt under its own
    id, and no owner message in the chat is labelled with the agent's act.
    """
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
    issuer = _routing_issuer(ctx)
    if issuer["kind"] == ISSUER_TASK:
        return _send_task_message(ctx, issuer, target, msg, current_chat_id)
    _md = getattr(ctx, "task_metadata", None)
    _md = _md if isinstance(_md, dict) else {}
    client_message_id = str(_md.get("client_message_id") or "").strip()
    # The model chooses the target, and the host transports the exact owner
    # bytes captured at ingress on the turn's FIRST routing act while it is
    # still acting on the message it was started with: a model-authored
    # paraphrase must not replace the owner's own steering text.  Two typed
    # facts end that window, and after either one the turn is RELAYING its own
    # words rather than paraphrasing the owner's — a later owner message the
    # turn actually DRAINED from its mailbox (``ctx.last_owner_delivery``,
    # stamped at the loop's drain seam), or its origin message already carried
    # somewhere by a landed routing receipt.  A message merely WRITTEN to the
    # mailbox has not reached the turn and ends nothing: the write-time counter
    # this used to read closed the window against a turn that was still acting
    # on its own origin, and took the steer's receipt with it.  Replaying the
    # origin past either real point re-sent a twenty-minute-old message and
    # silently dropped what the turn actually had to say (#896).  So the model's
    # text stands, and the receipt follows the owner message actually relayed —
    # the drained entry's own client id — or, for an agent-authored steer that
    # belongs to no owner message, the steer's OWN synthetic id: never again an
    # origin message this turn has already routed.  The receipt channel is keyed
    # by message id only because that is how `routing_wait` polls it, so silence
    # there would report a successful delivery as STEER_UNCONFIRMED and invite
    # the model to retry a message that landed.  Nothing in any chat carries a
    # synthetic id, so no owner message is labelled by the agent's own act.
    # Non-owner/internal calls have no origin text and retain the explicit
    # tool argument.
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    routing_token = uuid.uuid4().hex
    agent_authored_receipt_id = f"{AGENT_RECEIPT_ID_PREFIX}{routing_token}"
    delivery = getattr(ctx, "last_owner_delivery", None)
    if isinstance(delivery, dict) and delivery:
        # The turn is relaying its own words after a later owner message reached it.
        client_message_id = (
            str(delivery.get("client_message_id") or "").strip() or agent_authored_receipt_id
        )
    elif _origin_already_routed(ctx, client_message_id):
        client_message_id = agent_authored_receipt_id
    elif isinstance(_md.get("origin_message_text"), str):
        exact_owner_text = str(_md.get("origin_message_text") or "")
        if exact_owner_text.strip():
            msg = exact_owner_text
    # A task-authored steer belongs to no owner message at all (a managed task or
    # a project root has no chat ingress id), and the receipt channel is keyed by
    # message id ONLY because that is how ``routing_wait`` polls it. Left empty,
    # the supervisor wrote no annotation and the wait answered
    # `client_message_id_missing` before the handler even ran, so EVERY such steer
    # came back STEER_UNCONFIRMED — including the ones the host had already
    # refused in writing. The steer's own synthetic id gives it a receipt of its
    # own; it addresses no message in any chat, so nothing the owner wrote is
    # labelled with the agent's act (``_relayed_owner_message`` relays nothing
    # under this prefix, and compaction bounds these rows by their own cap).
    client_message_id = client_message_id or agent_authored_receipt_id
    evt: Dict[str, Any] = {
        "type": "steer_task",
        "routing_token": routing_token,
        "target_task_id": target,
        "message": msg,
        "chat_id": current_chat_id,
        "client_message_id": client_message_id,
        # The issuer fact by value: the supervisor keys the room veto, the
        # owner acknowledgement and the mailbox kind on it, never on a chat id
        # or an empty client id.
        "issuer": issuer,
        "attachment_uploads": list(_md.get("chat_attachment_uploads") or []),
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
    return _steer_refusal_text(target, mode, receipt)


def _steer_refusal_text(target: str, mode: str, receipt: Dict[str, Any]) -> str:
    """The typed refusal/unconfirmed sentence for one steer receipt."""
    status = str(receipt.get("status") or "unconfirmed")
    if status in {"rejected", "needs_manual_target"}:
        return (
            f"⚠️ STEER_REJECTED: task {target} was not steered "
            f"({str(receipt.get('reason') or 'target_not_steerable')})."
        )
    # Only "no receipt exists yet" reaches here: a settled refusal is returned
    # above with its reason, so UNCONFIRMED never disguises a known rejection.
    return (
        f"⚠️ STEER_UNCONFIRMED: mailbox delivery to task {target} was not durably confirmed "
        f"({mode}, {str(receipt.get('reason') or 'confirmation_timeout')}). "
        "Do not report the message as delivered."
    )


def _send_task_message(
    ctx: ToolContext, issuer: Dict[str, Any], target: str, msg: str, current_chat_id: int,
) -> str:
    """A task's own words to another host-listed root, never as owner text.

    The event rides the same supervisor rail as an owner steer (the target is
    revalidated live under the queue lock, the receipt is token-bound), keyed
    under the drained owner message's id when this round relays one, otherwise
    under a synthetic receipt id. Receipt identity never changes authorship.
    No origin-bytes substitution, attachments or owner client surface.
    The result says WRITTEN: the target reads it at its next checkpoint.
    """
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_sender_origin
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    routing_token = uuid.uuid4().hex
    delivery = getattr(ctx, "last_owner_delivery", None)
    owner_message_id = str(delivery.get("client_message_id") or "").strip() if isinstance(delivery, dict) else ""
    evt: Dict[str, Any] = {
        "type": "steer_task",
        "routing_token": routing_token,
        "target_task_id": target,
        "message": msg,
        "chat_id": current_chat_id,
        "client_message_id": owner_message_id or f"{AGENT_RECEIPT_ID_PREFIX}{routing_token}",
        "issuer": dict(issuer),
        "ts": utc_now_iso(),
    }
    if (binding := presence_caller_binding(ctx)) is not None:  # admitted only to this binding's own work (owner Q2)
        evt.update(presence_binding_id=binding, sender_origin=presence_sender_origin(ctx))
    mode, receipt = _emit_and_wait_for_routing(ctx, evt)
    if str(receipt.get("status") or "") == "delivered":
        return (
            f"✉️ Message to task {target} written to its mailbox (durably confirmed, {mode}). "
            "It reads it at its next checkpoint as a message from this task, not as owner "
            "text. Files cannot be attached to messages between tasks."
        )
    return _steer_refusal_text(target, mode, receipt)
