"""Tell another work: model-authored note plus selected host-held owner quotes."""
from __future__ import annotations

import uuid


def _teller_room(ctx) -> int:
    """Where a note without owner quotes returns: the teller's room by the room rule now (a running task
    turned into a Project after it started keeps its admitted chat until readmission), else its chat;
    a teller carrying no chat returns to Main."""
    from ouroboros.projects_registry import project_chat_for_task_tree
    from ouroboros.tool_access import canonical_data_root

    meta = getattr(ctx, "task_metadata", None) or {}
    room = project_chat_for_task_tree(canonical_data_root(ctx), str(getattr(ctx, "task_id", "") or ""),
                                      meta.get("parent_task_id", ""), meta.get("root_task_id", ""))
    own = getattr(ctx, "current_chat_id", None)
    return int(room or (1 if own is None else own))


def _hand_arrival(ctx, target: str, selector) -> str:
    """One captured message this place focus received, to one work of its conversation: ``""``, or the refusal's reason."""
    from ouroboros.place_intake import hand_own_arrival
    from ouroboros.tool_access import canonical_data_root

    return hand_own_arrival(canonical_data_root(ctx), getattr(ctx, "task_metadata", None), target, selector,
                            str(getattr(ctx, "task_id", "") or ""))


def _tell_task(ctx, task_id: str, note: str = "", quote: list | None = None, summary: str = "", arrival: bool | str = False) -> str:
    from ouroboros.owner_relay import quote_rows
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_sender_origin
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX
    from ouroboros.tools.control_routing import _routing_issuer, _emit_and_wait_for_routing, _attach_client_surface
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, publish_no_effect
    from ouroboros.utils import utc_now_iso

    target, handed = str(task_id or "").strip(), ""
    if arrival is not False:
        selector = arrival if arrival is True else arrival.strip() if isinstance(arrival, str) else ""
        if not selector or quote or not target:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", meta={"reason_code": "argument_invalid"},
                text="TOOL_ARG_ERROR: tell_task with arrival names one task_id, takes no quote, and arrival is true "
                     "or the source_event_id of a later message delivered to this focus."))
        try:
            refusal = _hand_arrival(ctx, target, selector)
        except OSError as exc:  # failure may precede the claim; never invent an exclusive holder
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                meta={"reason_code": "arrival_effect_unknown"},
                text=f"TELL_UNCONFIRMED: {target}: handing this message over may or may not have landed ({exc}); "
                     "any recorded recipient remains binding. Retry the same arrival to resolve it. The note was not sent."))
        if refusal:
            holder = refusal.partition(":")[2]  # this call wrote nothing; what an earlier one did stays with that holder
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE", meta={"reason_code": refusal},
                text=f"TELL_REJECTED: {target}: {refusal}. "
                     + (f"This call wrote nothing; the message stays with {holder}." if holder else "Nothing was delivered.")))
        handed = f"The message was handed to {target} exactly as captured; a durable write, not proof it was read."
        if not isinstance(note, str) or not note.strip():
            return handed
    quotes, error = quote_rows(ctx, [] if quote is None else quote)
    if error or not target or not isinstance(note, str) or (not note.strip() and not quotes):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", meta={"reason_code": "QUOTE_NOT_HELD" if error else "argument_invalid"},
            text=error or "TOOL_ARG_ERROR: tell_task needs a task_id and a note or held owner quote."))
    relay_id = uuid.uuid4().hex
    message_id = next((q["source"]["client_message_id"] for q in quotes if q["source"]["client_message_id"]), "")
    event = {"type": "steer_task", "routing_token": relay_id, "relay_id": relay_id,
             "target_task_id": target, "message": note, "quotes": quotes,
             "summary": summary.strip() if isinstance(summary, str) else "",
             "chat_id": int(quotes[0]["source"]["chat_id"] if quotes else _teller_room(ctx)),
             "client_message_id": message_id or f"{AGENT_RECEIPT_ID_PREFIX}{relay_id}",
             "issuer": {**_routing_issuer(ctx), "task_id": str(getattr(ctx, "task_id", "") or "")},
             "ts": utc_now_iso()}
    _attach_client_surface(ctx, event)
    if (binding := presence_caller_binding(ctx)) is not None:
        event.update(presence_binding_id=binding, sender_origin=presence_sender_origin(ctx))
    mode, receipt = _emit_and_wait_for_routing(ctx, event)
    status = receipt.get("status")
    if status in {"delivered", "read"}:
        return (f"{handed} " if handed else "") + f"Message written to {target}; relay_id={relay_id}. Read acknowledgement follows when it drains."
    reason = receipt.get("reason") or ("target_not_steerable" if status in {"rejected", "needs_manual_target"}
                                      else "confirmation_timeout")
    if status in {"rejected", "needs_manual_target"} and not handed:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
            text=f"TELL_REJECTED: {target}: {reason}. Nothing was delivered.", meta={"reason_code": reason}))
    if status in {"rejected", "needs_manual_target"}:  # two outcomes: the handover stands, only the note failed
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
            text=f"{handed} Its separate note was not delivered: TELL_REJECTED: {target}: {reason}.",
            meta={"reason_code": reason, "arrival_handed": True}))
    return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
        text=((f"{handed} Its separate note: " if handed else "") + f"TELL_UNCONFIRMED: {target}: {reason} ({mode}); "
              f"relay_id={relay_id}. The request may still land. Check its receipt before sending the words again."),
        meta={"reason_code": reason, "relay_id": relay_id, **({"arrival_handed": True} if handed else {})}))
