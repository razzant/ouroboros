"""Where a chat turn becomes a task, and where a project scope is bound.

Owns the routing acknowledgement the decision actor reads, the off-loop
preparation of the promoted source, the durable rejection record, the
rollback of a promoted task that never reached the queue, and the registry
side of project scope and per-cycle project digests.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from ouroboros.dialogue_provenance import presence_root_carrier
from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def _events():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import events

    return events


def _routing_project_address(ctx: Any, target: str, status: str) -> Dict[str, Any]:
    """Navigation address from the successful destination's registered binding."""
    if not target or status not in {"scheduled", "delivered"}:
        return {}
    try:
        from ouroboros.projects_registry import get_project, project_binding_for_task

        binding = project_binding_for_task(ctx.DRIVE_ROOT, target) or {}
        project = get_project(ctx.DRIVE_ROOT, str(binding.get("project_id") or ""))
        if project and project.get("chat_id") is not None and project.get("lifecycle") not in {"deleting", "deleted"}:
            return {"project_id": project["id"], "project_chat_id": int(project["chat_id"])}
    except Exception:
        log.debug("Routing destination projection unavailable", exc_info=True)
    return {}


def _routing_notice_carried(status: str, cause: str, options: Any, attachment_manifest: Any) -> bool:
    """Whether an ACK carries an owner notice: a refusal's cause, a picker or a file not attached."""
    rejected = any(isinstance(row, dict) and row.get("status") == "rejected" for row in attachment_manifest or [])
    return bool(cause) or rejected or (status == "needs_manual_target" and isinstance(options, list) and bool(options))


def _owner_message_receipt(client_message_id: Any) -> str:
    """The owner message this receipt id names, or "" for an act that wears none: no id, or the synthetic id of an
    act tied to no owner message (``AGENT_RECEIPT_ID_PREFIX``). Such an id annotates no chat row."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    client_message_id = str(client_message_id or "").strip()
    return "" if client_message_id.startswith(AGENT_RECEIPT_ID_PREFIX) else client_message_id


def _receipt_room(evt: dict) -> int:
    from ouroboros.chat_annotations import receipt_source_room
    from ouroboros.project_dialogue import owner_message_ref_is_valid
    ref = evt.get("source_ref")
    if evt.get("owner_source") and owner_message_ref_is_valid(ref) and ref["client_message_id"] == str(evt.get("client_message_id") or ""):
        return int(ref["chat_id"])
    try:
        return int(receipt_source_room(ref, _owner_message_receipt(evt.get("client_message_id")), evt.get("chat_id")) or 0)
    except (ValueError, TypeError):
        return 0


def _routing_notice_reply(ctx: Any, evt: Dict[str, Any], chat_id=None, *, address=None) -> Dict[str, Any]:
    """The owner message a routing notice answers, as its exact host Reply reference.

    The retained owner row the act quotes for this receipt, else the owner source it
    carries by value, else the accepted row its receipt id names. ``{}`` when none
    resolves (a synthetic act, an unreadable store): an explicit "no owner row", which
    keeps a task-keyed System row from borrowing its target task's own origin.
    ``address`` receives where the notice answers that row when it is an owner-bot Reply
    whose room the host decided (``chat_references.decided_reply_address``).
    """
    from ouroboros.chat_chain import _row_chat_id, resolve_row
    from ouroboros.chat_references import decided_reply_address, reference_from_row, visible_source_reply
    from ouroboros.owner_relay import owner_source_row
    from ouroboros.project_dialogue import owner_message_ref_is_valid, unique_inbound_message

    client_message_id = str(evt.get("client_message_id") or "").strip()
    receipt_room = _receipt_room(evt)
    destination = receipt_room if chat_id is None else chat_id
    try:
        source = evt.get("source_ref")
        quote = next((quote for quote in evt.get("quotes") or []
                      if client_message_id and isinstance(quote, dict) and isinstance(quote.get("owner_row"), dict)
                      and _row_chat_id(quote["owner_row"]) == receipt_room
                      and str((quote.get("source") or {}).get("client_message_id") or "") == client_message_id), None)
        row = quote["owner_row"] if quote is not None else None
        if not row or evt.get("owner_source"):
            if owner_message_ref_is_valid(source) and (not client_message_id or (
                    source["client_message_id"] == client_message_id and source["chat_id"] == receipt_room)):
                snapshot = evt.get("origin_message_snapshot") or {}
                reply = snapshot.get("reply") if snapshot.get("projectable") is True else None
                return visible_source_reply(ctx.DRIVE_ROOT, source, destination, reply=reply)
            if evt.get("owner_source") or evt.get("origin_suppressed") is True or (isinstance(source, dict) and source):
                return {}  # a source the host named but cannot show, or suppressed, is never guessed from the client id
        if row:
            root = row.get("_owner_source_root") or ctx.DRIVE_ROOT
            row = owner_source_row(row)
            source_address = quote["source"].get("address")
            if source_address and source_address != reference_from_row(row, root)["address"]:
                row, _ = resolve_row(root, source_address)
                if not row or _row_chat_id(row) != receipt_room:
                    return {}
            if address is not None:
                address.update(decided_reply_address(row))
            return visible_source_reply(root, None, destination, row=row)
        if _owner_message_receipt(client_message_id):
            row = unique_inbound_message(ctx.DRIVE_ROOT, {receipt_room}, client_message_id)
        if row and address is not None:
            address.update(decided_reply_address(row))
        return visible_source_reply(ctx.DRIVE_ROOT, None, destination, row=row) if row else {}
    except Exception:
        log.debug("Routing notice origin unavailable for %s", client_message_id, exc_info=True)
        return {}


def _routing_notice_ack_fields(ctx: Any, evt: Dict[str, Any], fields: Dict[str, Any], *,
                               notice_transport: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What an ACK sent from these receipt fields says of the act's owner notice: ``{}`` for a quiet ACK; else the
    carrier the receipt selects. That is the ACK itself, with the owner message it answers, when its id names one.
    An ACK whose id names none annotates no chat row, so it is never the carrier: it names ``system``, the System
    row its act's path writes where the host owns the notice (a host-issued act, a steer of my human's words) and
    every client shows; a tool act tied to no owner message writes none, and the model narrates its tool result.
    An ACK answering an owner-bot Reply whose room the host decided also names that Reply's own ``transport``.
    No field is a row field, and none proves that a transport delivered anything."""
    if not _routing_notice_carried(fields["status"], str(fields.get("cause") or ""), fields.get("options"),
                                   fields.get("attachment_manifest")):
        return {}
    message = _owner_message_receipt(evt.get("client_message_id"))
    unplaced = _receipt_unplaced(fields)
    if not message or ((unplaced or message.startswith("quiz_answer:")) and fields.get("cause")
                       and _system_row_follows(evt, fields)):
        # No chat bubble wears an unplaced receipt or an answered quiz's hidden answer row: where the act's own path
        # writes the System row its receipt selects, that row carries the notice, never a second standalone one.
        return {"routing_notice_carrier": "system"}
    address: Dict[str, Any] = dict(notice_transport or {})
    reply = {} if unplaced else _routing_notice_reply(ctx, evt, address=None if address else address)
    return {"routing_notice_carrier": "routing_ack", "reply_to": reply, **({"transport": address} if address else {})}


def _receipt_unplaced(fields: Dict[str, Any]) -> bool:
    """A receipt whose host stated that no row wears it (`chat_annotations.explicit_source_fields`)."""
    return "source_row_address" in fields and not fields["source_row_address"]


def _system_row_follows(evt: Dict[str, Any], fields: Dict[str, Any]) -> bool:
    """Whether the act's own path publishes the System row its receipt selects as the notice's carrier: an owner's
    steer (`steering._refusal_row` when `_notice_row_owed`) and a host-issued promote (`_notify_host_initiated_refusal`).
    Only a refusal with a cause reaches either writer, so an attachment-only ACK of a started act keeps its notice, and
    so does a task-issued act (it reads its typed refusal in its tool result)."""
    status = fields.get("status")
    if fields.get("action") == "steer_task":
        issuer = evt.get("issuer")
        return status in {"rejected", "needs_manual_target"} and not (isinstance(issuer, dict) and issuer.get("kind") == "task")
    return bool(evt.get("host_initiated")) and status in {"needs_manual_target", "unconfirmed"}


# Acts that pass the owner's words on to a running work: their receipt names that work and its room.
_PASSED_ON_ACTS = frozenset({"steer_task", "mailbox_delivery"})


def _routing_receipt_fields(
    ctx: Any,
    evt: Dict[str, Any],
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    reason: str = "",
    options: Optional[list] = None,
    attachment_manifest: Optional[list] = None,
    reasoning_effort: str = "",
    unnamed_label: str = "Task",
    **supplied: Any,
) -> Dict[str, Any]:
    """The fields one routing receipt states, read once (#968).

    The supervisor's token-bound receipts and the server's direct one write
    their durable row and send their live frame from this one dict, so a second
    read can no longer make the two disagree. Words passed on to a running work
    name that work and its room (the relay's facts): a relay names the room as it
    always did, any other act only an ACTIVE Project it landed in. Another landed
    act carries the Project address of its registered binding. A supervisor relay
    whose work cannot be read fails as it always did; for any other act the work's
    facts are added enrichment, so the receipt falls back to that address rather
    than being lost. ``supplied`` is the caller's own presentation, forwarded as
    given and never chosen here; only ``ROUTING_RECEIPT_FIELDS`` reach the durable row. ``unnamed_label``: a nameless
    work's label ("" for a Main turn's saved words: no placeholder name).
    """
    from ouroboros.project_dialogue import routing_refusal_cause, routing_target_label

    if target and not str(target_label or "").strip():
        target_label = routing_target_label(ctx.DRIVE_ROOT, action, target, task=evt, project_id=str(evt.get("project_id") or ""),
                                            fallback=unnamed_label)
    fields: Dict[str, Any] = {**supplied, "action": action, "target": target, "target_label": target_label, "status": status}
    from ouroboros.chat_annotations import routing_source_address
    if "source_row_address" not in fields and (address := routing_source_address(
            evt.get("source_ref"), _receipt_room(evt), evt.get("client_message_id"))):
        fields["source_row_address"] = address
    optional = {
        "routing_token": str(evt.get("routing_token") or "").strip(),
        # Q3=A: the owner-facing sentence for a REFUSED act ("" for a landed row or the picker).
        "cause": routing_refusal_cause(action, status, reason, options),
        "reasoning_effort": str(reasoning_effort or ""),
    }
    fields.update({key: value for key, value in optional.items() if value})
    if options is not None:
        fields["options"] = options
    if attachment_manifest is not None:
        fields["attachment_manifest"] = attachment_manifest
    if not (target and action in _PASSED_ON_ACTS):
        fields.update(_routing_project_address(ctx, target, status))
        return fields
    from ouroboros.owner_relay import relay_target_facts

    try:
        facts = relay_target_facts(ctx.DRIVE_ROOT, target, evt.get("relay_target"))
    except Exception:
        if evt.get("relay_id"):
            raise  # before any row is written, as a relay's receipt always failed
        log.debug("Routing work facts unavailable for %s", target, exc_info=True)
        fields.update(_routing_project_address(ctx, target, status))
        return fields
    if not evt.get("relay_id") and not _routing_room_open(ctx, str(facts.get("project_id") or ""), status):
        # The work keeps its name; a deleting or tombstoned room is no address to open.
        facts = {key: value for key, value in facts.items() if key not in ("project_id", "project_chat_id")}
    fields.update(facts)
    return fields


def _routing_room_open(ctx: Any, project_id: str, status: str) -> bool:
    """Whether a landed act's receipt may open this Project's room: an ACTIVE one only (``_routing_project_address``)."""
    if not project_id or status not in {"scheduled", "delivered"}:
        return False
    try:
        from ouroboros.projects_registry import get_project

        return get_project(ctx.DRIVE_ROOT, project_id) is not None
    except Exception:
        log.debug("Routing room lifecycle unavailable for %s", project_id, exc_info=True)
        return False


def _emit_routing_receipt(
    ctx: Any,
    evt: Dict[str, Any],
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    reason: str = "",
    detail: str = "",
    options: Optional[list] = None,
    attachment_manifest: Optional[list] = None,
    publish: bool = True,
    reasoning_effort: str = "",
    source_row_address: str = "",
) -> Dict[str, Any]:
    """Persist and publish one token-bound routing annotation receipt."""
    fields = _routing_receipt_fields(
        ctx, evt, action=action, target=target, target_label=target_label, status=status, reason=reason,
        options=options, attachment_manifest=attachment_manifest, reasoning_effort=reasoning_effort,
        **({"source_row_address": source_row_address} if source_row_address else {}),
    )
    client_message_id = str(evt.get("client_message_id") or "").strip()
    # An act that named a source (a reference of any shape, a selected owner source, the door's suppression) states
    # where it landed ONCE: the exact address, or `source_row_address: ""` for what the host cannot verify. Its row
    # and its live frame carry that one fact, so no reload and no mounted bubble places it by client id.
    from ouroboros.chat_annotations import explicit_source_fields
    fields.update(explicit_source_fields(
        ctx.DRIVE_ROOT, evt.get("source_ref"), _receipt_room(evt), _owner_message_receipt(client_message_id), fields,
        suppressed=evt.get("origin_suppressed") is True, selected=bool(evt.get("owner_source")), picker=bool(options)))
    unplaced = _receipt_unplaced(fields)
    annotation_status = "not_applicable"
    if client_message_id:
        try:
            from ouroboros.project_dialogue import ROUTING_RECEIPT_FIELDS, append_chat_annotation

            annotation_status = (
                "persisted"
                if append_chat_annotation(
                    ctx.DRIVE_ROOT,
                    client_message_id,
                    source_chat_id=_receipt_room(evt),
                    reason=reason,
                    detail=detail,
                    **({"source_unplaced": True} if unplaced else {}),
                    **{key: value for key, value in fields.items() if key in ROUTING_RECEIPT_FIELDS},
                )
                else "failed"
            )
        except Exception:
            annotation_status = "failed"
            log.debug("Routing annotation append failed", exc_info=True)

    effective_status = str(status or "needs_manual_target")
    effective_reason = str(reason or "")
    if annotation_status == "failed" and effective_status in {"scheduled", "delivered"}:
        effective_status = "unconfirmed"
        effective_reason = "routing_annotation_persist_failed"

    receipt: Dict[str, Any] = {
        # True only when a row was actually written: an event that carries no
        # receipt id has no receipt, and saying "persisted" for it is how an
        # ensure_project_scope handler reported a bind nobody could read.
        # ``annotation_status`` keeps the three-way fact for callers whose
        # positive authority lives elsewhere (a promote's admission record).
        "persisted": annotation_status == "persisted",
        "status": effective_status,
        "reason": effective_reason,
        "detail": str(detail or ""),
        "annotation_status": annotation_status,
        "routing_token": str(fields.get("routing_token") or ""),
        "target_label": str(fields["target_label"] or ""),
        # What the row was written from: a publish deferred past the admission record sends exactly this.
        "fields": fields,
    }
    if attachment_manifest is not None:
        receipt["attachment_manifest"] = _events()._routing_attachments(attachment_manifest) or []
    # The carrier SELECTED for the act's owner notice (never proof that a transport
    # delivered it): the ACK when one carrying it was handed over and answers a chat row;
    # for a known refusal without such an ACK, the System row its path writes, if any;
    # "" when there is nothing to say.
    receipt["routing_notice_carrier"] = "system" if fields.get("cause") and publish else ""
    if annotation_status == "failed":
        # Fail closed: no ACK outruns its durable row, so a landed act reads unconfirmed
        # and a picker is never clickable without its authority.
        return receipt
    if publish:
        carrier = _publish_routing_ack(ctx, evt, {**fields, "status": effective_status})
        receipt["routing_notice_carrier"] = carrier or receipt["routing_notice_carrier"]
    return receipt


def _publish_routing_ack(ctx: Any, evt: Dict[str, Any], fields: Dict[str, Any]) -> str:
    """Publish a live non-bubble acknowledgement after durable authority exists:
    the fields its row was written from (``_routing_receipt_fields``), never a second read.

    An ACK carrying the act's owner notice also names the carrier its receipt selects:
    itself and the owner message it answers, or the System row when it answers no chat
    row (``_routing_notice_ack_fields``, #966). The return is that carrier once the bus
    took the ACK without raising ("" for a quiet ACK or none sent).
    """
    try:
        chat_id = _receipt_room(evt)
        ack = getattr(getattr(ctx, "bridge", None), "send_routing_ack", None)
        if callable(ack):
            notice = _routing_notice_ack_fields(ctx, evt, fields, notice_transport=_promotion_notice_transport(ctx, evt))
            ack(chat_id, client_message_id=str(evt.get("client_message_id") or "").strip(), **fields, **notice)
            return str(notice.get("routing_notice_carrier") or "")
    except Exception:
        log.debug("Routing typed ack failed", exc_info=True)
    return ""


def _promotion_notice_transport(ctx: Any, evt: Dict[str, Any]) -> dict:
    """Use the supervisor event's admitted turn address even when its quote cannot be read."""
    from ouroboros.chat_references import TURN_ADDRESS, reply_transport_from_metadata

    return reply_transport_from_metadata(ctx.DRIVE_ROOT, evt, anchor=False) if TURN_ADDRESS in evt else {}


def _handle_project_digest(evt: Dict[str, Any], ctx: Any) -> None:
    """A project task finished: touch the project.

    Full project awareness (v6.32.0): the one identity already sees the project's
    chat thread in its unified memory and the finished task in ``task_results``,
    so the digest is neither a message nor a wake: the next SCHEDULED wake-up
    renders the facts itself (``consciousness_wake``) and the one agent decides
    what to do with them — backlog, identity, or nothing (BIBLE P5). Events never
    move the wake (owner 12 Oct, #1603).
    """
    pid = str(evt.get("project_id") or "").strip()
    if not pid:
        return
    try:
        from ouroboros.projects_registry import touch_project

        touch_project(ctx.DRIVE_ROOT, pid)
    except Exception:
        log.debug("project_digest touch failed", exc_info=True)


def _rollback_promoted_pending(
    ctx: Any, task_id: str, admission_token: str, *, reason: str,
) -> bool:
    """Remove an unconfirmed promote before the supervisor can assign it."""
    from supervisor import queue as supervisor_queue

    removed = False
    with supervisor_queue._queue_lock:
        pending = getattr(ctx, "PENDING", supervisor_queue.PENDING)
        survivors = [
            task for task in pending
            if not (
                str(task.get("id") or "") == task_id
                and str(
                    task.get("_admission_owner_token")
                    or task.get("promotion_admission_token")
                    or ""
                ) == admission_token
            )
        ]
        removed = len(survivors) != len(pending)
        if removed:
            pending[:] = survivors
    if removed:
        persist = getattr(ctx, "persist_queue_snapshot", None)
        if callable(persist):
            try:
                persist(reason=reason)
            except Exception:
                log.warning("Failed to persist promote rollback for %s", task_id, exc_info=True)
    return removed


def _own_emitted_stub(ctx: Any, task_id: str, routing_token: str) -> bool:
    """Whether the row on disk is THIS promote's emitted pre-receipt (#1160). A
    refusal that writes no result of its own must still replace it, or the
    reconciliation read answers "admission pending" for ever."""
    try:
        from ouroboros.routing_wait import is_own_admission_stub
        from ouroboros.task_results import load_task_result

        return is_own_admission_stub(load_task_result(ctx.DRIVE_ROOT, task_id), routing_token)
    except Exception:
        log.warning("promote: emitted-stub lookup failed for %s", task_id, exc_info=True)
        return False


def _persist_promote_rejection(
    ctx: Any,
    evt: Dict[str, Any],
    outcome: Dict[str, Any],
    *,
    status: str = "rejected",
) -> None:
    task_id = str(outcome.get("task_id") or evt.get("task_id") or "")
    reason = str(outcome.get("reason") or "admission_rejected")
    if reason == "task_id_lookup_failed":
        return  # preserve the unreadable exact-id authority byte-for-byte
    from supervisor.task_admission import persist_never_admitted_refusal

    never_admitted = bool(outcome.get("never_admitted") and status == "rejected")
    cleanup_manifest = outcome.pop("_admission_cleanup_manifest", None)
    write_refusal = persist_never_admitted_refusal if never_admitted else write_task_result
    write_refusal(
        ctx.DRIVE_ROOT,
        task_id,
        **({"admission_token": str(evt.get("routing_token") or "")}
           if never_admitted else {"status": STATUS_FAILED}),
        reason_code=reason,
        project_id=str(evt.get("project_id") or ""),
        description=str(evt.get("objective") or ""),
        expected_output=str(evt.get("expected_output") or ""),
        promotion_admission={
            "status": status,
            "routing_token": str(evt.get("routing_token") or ""),
            "reason": reason,
            "detail": str(outcome.get("detail") or ""),
            "worker_pool_disabled_reason": str(
                outcome.get("worker_pool_disabled_reason") or ""
            ),
            "confirmed_at": utc_now_iso(),
        },
        result=(
            f"Promotion was not scheduled: {reason}. "
            f"{str(outcome.get('detail') or '')}"
        ).strip(),
    )
    if never_admitted:
        from ouroboros.artifacts import remove_staged_attachments
        from ouroboros.headless import remove_subagent_task_drive
        from supervisor.queue import task_settlement_interlock, task_settlement_liveness

        if cleanup_manifest:
            remove_staged_attachments(cleanup_manifest)
        remove_subagent_task_drive(ctx.DRIVE_ROOT, task_id, live=task_settlement_liveness,
                                   guard=task_settlement_interlock, admission_rollback=True)


def _prepare_promote_source_off_loop(evt: Dict[str, Any], ctx: Any) -> None:
    """Resolve a potentially 900s clone away from the supervisor drain loop."""
    continuation = dict(evt)
    continuation["_source_prepared"] = True
    try:
        from ouroboros.promotion_source import resolve_promote_source

        source_basis = {}
        folder, note, error, project_id, source_created = resolve_promote_source(
            ctx,
            str(evt.get("source") or ""),
            str(evt.get("project_id") or ""),
            project_name=str(evt.get("project_name") or ""), admission_basis_out=source_basis,
        )
        if source_basis:
            continuation["_project_admission"] = source_basis
        continuation["project_id"] = project_id
        continuation["_source_note"] = note
        continuation["_source_error"] = error
        continuation["_source_created"] = bool(source_created)
        if folder and not str(continuation.get("workspace_root") or "").strip():
            continuation["workspace_root"] = folder
    except Exception as exc:
        continuation["_source_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from supervisor.workers import get_event_q

        get_event_q().put(continuation)
    except Exception as exc:
        log.exception("Failed to publish promote source continuation")
        from supervisor import queue as supervisor_queue

        task_id = str(evt.get("task_id") or "")
        routing_token = str(evt.get("routing_token") or "")
        supervisor_queue.release_task_admission(task_id, routing_token)
        # No host producer (skill card, Swarm, picker click) stamps `source`, so
        # this exit never bypasses the wrapper's host-initiated refusal notice.
        failed = {
            "status": "unconfirmed",
            "reason": "source_continuation_publish_failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "task_id": task_id,
        }
        try:
            _persist_promote_rejection(ctx, evt, failed, status="unconfirmed")
            _emit_routing_receipt(
                ctx,
                evt,
                action=(
                    "route_to_project"
                    if bool(evt.get("routed_from_main"))
                    else "promote_chat_to_task"
                ),
                target=task_id,
                status="unconfirmed",
                reason=failed["reason"],
                detail=failed["detail"],
            )
        except Exception:
            log.exception("Failed to persist promote source continuation failure")


def _promote_chat_to_task_outcome(evt: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Spawn a first-class pooled owner task from a conversation-lane promote.

    Unlike ``schedule_subagent`` the child is NOT a subagent: it is a normal
    owner task (live card, canonical drive, project lease participation). The
    conversation lane that emitted the event stays free. Every exit returns the
    typed outcome to ``_handle_promote_chat_to_task``, the one publication
    boundary that tells the owner about a host-initiated refusal.
    """
    from supervisor.workers import (
        _broadcast_task_named,
        promote_chat_to_task,
        worker_pool_admission_state,
    )
    receipt_action = (
        "route_to_project" if bool(evt.get("routed_from_main"))
        else "promote_chat_to_task"
    )

    task_id = str(evt.get("task_id") or "")
    routing_token = str(evt.get("routing_token") or "")
    try:
        from supervisor import queue as supervisor_queue

        reservation = supervisor_queue.reserve_task_admission(
            task_id,
            routing_token,
            require_worker_pool=True,
            drive_root=ctx.DRIVE_ROOT,
            worker_pool=getattr(ctx, "WORKERS", None),
        )
        reservation_status = str(reservation.get("status") or "")
        if reservation_status == "already_reserved" and evt.get("_admission_reserved"):
            reservation_status = "reserved"
        if reservation_status != "reserved":
            if reservation_status == "existing_same_token":
                admission = reservation.get("promotion_admission")
                return {
                    "status": str((admission or {}).get("status") or "unconfirmed"),
                    "task_id": task_id,
                    "reason": str((admission or {}).get("reason") or ""),
                    # A replay of an already-settled admission: the owner was
                    # told once, so the refusal notice stays silent.
                    "replayed": True,
                }
            if reservation_status == "already_reserved":
                return {"status": "preparing", "task_id": task_id}
            blocked = {
                "status": "needs_manual_target",
                "reason": str(reservation.get("reason") or "admission_reservation_failed"),
                "worker_pool_disabled_reason": str(
                    reservation.get("worker_pool_disabled_reason") or ""
                ),
                "task_id": task_id,
                "reservation_owned": False,
            }
            if blocked["reason"] in {"worker_pool_unavailable", "worker_pool_state_unavailable"}:
                blocked["never_admitted"] = True
            if blocked["reason"] != "duplicate_task_id" or _own_emitted_stub(ctx, task_id, routing_token):
                _persist_promote_rejection(ctx, evt, blocked)
            blocked["_routing_notice_carrier"] = _emit_routing_receipt(
                ctx,
                evt,
                action=receipt_action,
                target=task_id,
                status="needs_manual_target",
                reason=blocked["reason"],
            )["routing_notice_carrier"]
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "promote_chat_to_task_rejected",
                    "task_id": task_id,
                    "reason": blocked["reason"],
                    "worker_pool_disabled_reason": blocked[
                        "worker_pool_disabled_reason"
                    ],
                },
            )
            return blocked
        from ouroboros.promotion_intent import claim_promotion_intent

        claim_promotion_intent(ctx.DRIVE_ROOT, evt)
        evt = {**evt, "_admission_reserved": True}
        from ouroboros.presence_owner_contact import private_start_refusal

        private_refusal = private_start_refusal(ctx.DRIVE_ROOT, evt)
        from supervisor.worker_promotion import _promotion_origin_error
        origin_error = _promotion_origin_error(evt)
        if not (origin_error or private_refusal) and str(evt.get("source") or "").strip() and not evt.get("_source_prepared"):
            threading.Thread(
                target=_prepare_promote_source_off_loop,
                args=(dict(evt), ctx),
                daemon=True,
                name=f"promote-source-{task_id[:12]}",
            ).start()
            return {"status": "preparing", "task_id": task_id}
        source_error = str(evt.get("_source_error") or "")
        if private_refusal or source_error or origin_error:
            outcome = {
                "status": "needs_manual_target",
                "reason": private_refusal or ("project_binding_failed" if origin_error else "project_source_error"),
                "never_admitted": True,
                "detail": origin_error or source_error,
                "task_id": task_id,
                "reservation_owned": True,
            }
        else:
            outcome = None
        pool_state = worker_pool_admission_state(ctx)
        if outcome is None and not pool_state["available"]:
            outcome = {
                "status": "needs_manual_target",
                "reason": "worker_pool_unavailable",
                "never_admitted": True,
                "worker_pool_disabled_reason": str(pool_state.get("disabled_reason") or ""),
                "task_id": task_id,
            }
        elif outcome is None:
            outcome = promote_chat_to_task(evt, ctx)
        outcome = outcome if isinstance(outcome, dict) else {"status": "scheduled"}
        admitted_task_contract = outcome.pop("_admitted_task_contract", None)
        admitted_metadata = outcome.pop("_admitted_metadata", None) or {}
        admitted_record = outcome.pop("_admitted_record", None) or {}
        if str(outcome.get("status") or "") == "scheduled":
            suggested_name = str(outcome.pop("_admitted_suggested_name", "") or "")
            receipt = _emit_routing_receipt(
                ctx,
                evt,
                action=receipt_action,
                target=str(outcome.get("task_id") or task_id),
                status="scheduled",
                detail=str(outcome.get("source_note") or ""),
                attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
                publish=False,
            )
            # The admission record is the positive authority; the annotation is
            # required only where an owner message exists to carry it.
            admission_status = (
                "scheduled"
                if str(receipt.get("annotation_status") or "") in {"persisted", "not_applicable"}
                and str(receipt.get("status") or "") == "scheduled"
                else "unconfirmed"
            )
            carrier = presence_root_carrier(evt, task_contract=evt.get("task_contract"))
            stored = write_task_result(
                ctx.DRIVE_ROOT,
                str(outcome.get("task_id") or task_id),
                STATUS_SCHEDULED,
                root_task_id=str(outcome.get("task_id") or task_id),
                delegation_role="root",
                task_contract=admitted_task_contract,
                **({} if carrier else admitted_record),  # room, folder, drive and title as admitted; Presence keeps its own
                **({"origin_message_ref": evt["source_ref"],
                    "origin_message_text": "" if (evt.get("origin_message_snapshot") or {}).get("placeholder_text")
                    else evt.get("source_text")} if evt.get("source_ref") else {}),
                **{k: v for k, v in outcome.items() if k == "_project_admission"},
                project_id=str(outcome.get("project_id") or evt.get("project_id") or ""),
                description=str(evt.get("objective") or ""),
                expected_output=str(evt.get("expected_output") or ""),
                suggested_name=suggested_name,
                # The explicit starting effort is admission authority: a root interrupted
                # before its worker's first write is Continued from THIS row.
                **({"reasoning_effort": evt["reasoning_effort"]} if evt.get("reasoning_effort") else {}),
                promotion_admission={
                    "status": admission_status,
                    "routing_token": str(evt.get("routing_token") or ""),
                    "reason": str(receipt.get("reason") or ""),
                    "confirmed_at": utc_now_iso(),
                    "queue_snapshot_persisted": True,
                    "routing_receipt_required": bool(str(evt.get("client_message_id") or "")),
                    "routing_receipt_status": str(receipt.get("annotation_status") or ""),
                    "source_note": str(outcome.get("source_note") or ""),
                },
                result=(
                    "Task accepted and durably scheduled."
                    if admission_status == "scheduled"
                    else "Task is scheduled, but its owner-facing routing receipt was not confirmed."
                ),
                attachment_manifest=list(outcome.get("attachment_manifest") or []),
                # Preserve start sources and Presence provenance even if this root never runs.
                **({"metadata": {**admitted_metadata, **carrier}} if admitted_metadata or carrier else {}),
                **({"source": "presence_promote"} if carrier else {}),
            )
            admission = stored.get("promotion_admission") if isinstance(stored, dict) else {}
            if (
                str((admission or {}).get("status") or "") != admission_status
                or str((admission or {}).get("routing_token") or "")
                != str(evt.get("routing_token") or "")
            ):
                raise RuntimeError("scheduled promotion result was not persisted")
            supervisor_queue.release_task_admission(task_id, routing_token)
            if admission_status != "scheduled":
                return {
                    **outcome,
                    "status": "unconfirmed",
                    "reason": str(receipt.get("reason") or "routing_receipt_persist_failed"),
                }
            from ouroboros.project_handoff import enqueue_project_handoff

            enqueue_project_handoff(ctx.DRIVE_ROOT, str(outcome.get("task_id") or task_id))
            _publish_routing_ack(ctx, evt, receipt["fields"])
            if suggested_name:
                _broadcast_task_named(
                    {"type": "task_named", "task_id": str(outcome.get("task_id") or task_id),
                     "suggested_name": suggested_name}
                )
            try:
                ctx.append_jsonl(
                    ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "promote_chat_to_task_admitted",
                        "task_id": str(outcome.get("task_id") or task_id),
                    },
                )
            except Exception:
                log.warning("Failed to record admitted promote %s", task_id, exc_info=True)
            return outcome

        if outcome.get("admission_started"):
            outcome["status"] = "unconfirmed"
            outcome["_routing_notice_carrier"] = _emit_routing_receipt(
                ctx, evt, action=receipt_action, target=task_id,
                status="unconfirmed", reason=str(outcome.get("reason") or ""),
                detail="Admission persistence is unconfirmed; task resources were retained.")["routing_notice_carrier"]
            return outcome
        _rollback_promoted_pending(
            ctx,
            str(outcome.get("task_id") or task_id),
            routing_token,
            reason="promote_chat_to_task_rejected",
        )
        supervisor_queue.release_task_admission(task_id, routing_token)
        if (outcome.get("never_admitted") or str(outcome.get("reason") or "") != "attachment_admission_rejected"
                or _own_emitted_stub(ctx, task_id, routing_token)):
            _persist_promote_rejection(ctx, evt, outcome)
        outcome["_routing_notice_carrier"] = _emit_routing_receipt(
            ctx,
            evt,
            action=receipt_action,
            target=str(outcome.get("task_id") or task_id),
            status="needs_manual_target",
            reason=str(outcome.get("reason") or "admission_rejected"),
            detail=str(outcome.get("detail") or ""),
            attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
        )["routing_notice_carrier"]
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_rejected",
                "task_id": str(outcome.get("task_id") or evt.get("task_id") or ""),
                "reason": str(outcome.get("reason") or "admission_rejected"),
                "project_lifecycle": str(outcome.get("project_lifecycle") or ""),
                "worker_pool_disabled_reason": str(
                    outcome.get("worker_pool_disabled_reason") or ""
                ),
            },
        )
        return outcome
    except Exception as exc:
        log.warning("promote_chat_to_task event failed", exc_info=True)
        # May have persisted or executed: never terminalize/clean an uncertain
        # attempt. A late exception after the scheduled receipt can be read back.
        from ouroboros.task_results import load_task_result

        try:
            stored = load_task_result(ctx.DRIVE_ROOT, task_id, strict=True) or {}
            admission = stored.get("promotion_admission") or {}
            if admission.get("routing_token") == routing_token and admission.get("status") == "scheduled":
                return {"status": "scheduled", "task_id": task_id, "replayed": True}
        except Exception:
            log.warning("Promotion receipt readback unavailable for %s", task_id, exc_info=True)
        failed_outcome = {
            "status": "unconfirmed", "reason": "promotion_persistence_failed", "task_id": task_id,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        failed_outcome["_routing_notice_carrier"] = _emit_routing_receipt(
            ctx,
            evt,
            action=receipt_action,
            target=str(evt.get("task_id") or ""),
            status="unconfirmed",
            reason="promotion_persistence_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )["routing_notice_carrier"]
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_failed",
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return failed_outcome


def _notify_host_initiated_refusal(ctx: Any, evt: Dict[str, Any], outcome: Any, carrier: str = "") -> None:
    """The ONE place a HOST-issued promote (skill card, Swarm, picker click)
    tells the owner it did not start (Q2=A). No model turn narrates such a
    refusal, so exactly one typed System row lands in the chat the OWNER wrote
    in — never ``task["chat_id"]``, which project admission may have rewritten
    to a new room — bound to the task that never started. A tool-issued promote
    gets nothing here (its receipt plus the failed call are the record and the
    model narrates); preparing, scheduled and replayed outcomes send nothing.
    The row names the carrier its act's receipt selected: ``routing_ack`` when the
    ACK on the owner's message already carried the notice (a transport showing ACKs
    skips this copy), otherwise ``system``. It replies to the owner's message, never
    to the unstarted task's own origin.
    """
    if not evt.get("host_initiated") or not isinstance(outcome, dict) or outcome.get("replayed"):
        return
    status = str(outcome.get("status") or "")
    if status not in {"needs_manual_target", "unconfirmed"}:
        return
    try:
        from ouroboros.project_dialogue import routing_refusal_cause
        from supervisor.message_bus import notification_chat_route

        chat = notification_chat_route(evt.get("chat_id"))
        if chat is None:
            return
        first_line = next(iter(str(evt.get("objective") or "").strip().splitlines()), "")
        if len(first_line) > 60:
            # An untitled act is named by its request's first words, cut at a
            # word boundary so the row never ends mid-word.
            first_line = first_line[:60].rsplit(" ", 1)[0].rstrip() + "…"
        title = str(evt.get("title") or evt.get("suggested_name") or "").strip() or first_line or "Task"
        action = "route_to_project" if bool(evt.get("routed_from_main")) else "promote_chat_to_task"
        reason = str(outcome.get("reason") or ("admission_rejected" if status == "needs_manual_target" else ""))
        address = _promotion_notice_transport(ctx, evt)
        progress_meta = {"routing_notice_carrier": "routing_ack" if carrier == "routing_ack" else "system",
                         "reply_to": _routing_notice_reply(ctx, evt, chat)}
        if address:
            progress_meta["transport"] = address
        ctx.send_with_budget(
            chat, f"{title} · {routing_refusal_cause(action, status, reason, None)}", role="system",
            system_type="task_start_unconfirmed" if status == "unconfirmed" else "task_not_started",
            task_id=str(outcome.get("task_id") or evt.get("task_id") or ""),
            progress_meta=progress_meta,
        )
    except Exception:
        log.warning("host-initiated refusal row failed for %s", evt.get("task_id"), exc_info=True)


def _handle_promote_chat_to_task(evt: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """The one publication boundary of a promote: every outcome of the handler
    body — the reservation-blocked and unconfirmed early returns included —
    passes the host-initiated refusal notice before it is returned."""
    outcome = _promote_chat_to_task_outcome(evt, ctx)
    carrier = outcome.pop("_routing_notice_carrier", "") if isinstance(outcome, dict) else ""
    _notify_host_initiated_refusal(ctx, evt, outcome, carrier)
    return outcome


def _handle_ensure_project_scope(evt: Dict[str, Any], ctx: Any) -> None:
    """Frozen receipt adapter for placement of the current work."""
    from supervisor.workers import ensure_project_scope

    try:
        outcome = ensure_project_scope(evt, ctx)
    except Exception as exc:
        log.warning("ensure_project_scope event failed", exc_info=True)
        outcome = {"status": "rejected", "reason": "ensure_project_scope_failed",
                   "detail": f"{type(exc).__name__}: {exc}"}
    if not isinstance(outcome, dict):
        outcome = {"status": "unconfirmed", "reason": "handler_returned_no_outcome"}
    target = str(outcome.get("project_id") or evt.get("project_id") or "")
    label = ""
    try:
        from ouroboros.projects_registry import get_project

        label = str((get_project(ctx.DRIVE_ROOT, target) or {}).get("name") or "") if target else ""
    except Exception:
        log.debug("ensure_project_scope: project name lookup failed", exc_info=True)
    _emit_routing_receipt(
        ctx, evt, action="ensure_project_scope", target=target, target_label=label or target,
        status=str(outcome.get("status") or "unconfirmed"),
        reason=str(outcome.get("reason") or ""), detail=str(outcome.get("detail") or ""),
        publish=False,
    )
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "ensure_project_scope_settled",
            "task_id": str(evt.get("task_id") or ""), "project_id": target,
            "status": str(outcome.get("status") or ""), "reason": str(outcome.get("reason") or ""),
            "routing_token": str(evt.get("routing_token") or ""),
        })
    except Exception:
        log.debug("ensure_project_scope settle row failed", exc_info=True)
