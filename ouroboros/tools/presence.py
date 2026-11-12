"""Inspect and edit places and their circles through the same writers as the owner's Circles page."""
from __future__ import annotations

import json
from typing import Any
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

# Frozen forced-finalization records still decode these wire outcomes.
PRESENCE_OUTCOMES = ("message", "silent", "tool_delivered", "deferred")


def _configure_presence(ctx: ToolContext, action: str, place_id: str = "", **params: Any) -> str:
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.owner_words import owner_words_contain
    from ouroboros import places
    from ouroboros.place_projection import circle_summary, places_view, place_view, connect_declared_place
    from ouroboros.tool_access import canonical_data_root
    root = canonical_data_root(ctx)
    metadata = getattr(ctx, "task_metadata", {}) or {}
    owner = (bool(getattr(ctx, "is_direct_chat", False)) and run_origin({"metadata": metadata})["owner_ingress"]
             and metadata.get("wake_cause") != "focus" and metadata.get("initiator") != "consciousness")
    words = params.get("description") if action == "set_description" else params.get("text")
    if action in {"add_note", "set_description"} or (action in {"create_circle", "edit_circle"} and words is not None):
        owner = owner and owner_words_contain(ctx, words)  # an act is ordered by the turn; a text is signed by its composer
    author = places.AUTHOR_OWNER if owner else places.AUTHOR_OUROBOROS
    try:
        if action == "list":
            result = places_view(root)
        elif action == "inspect":
            records = places.place_records(root)
            if place_id not in records.places:
                raise places.PlaceError("place_not_found", "place_id")
            result = {"place": place_view(root, records.places[place_id], records=records),
                      "circle": records.circle_context(place_id)}
        elif action == "connect":
            result = {"place": place_view(root, connect_declared_place(root, **params))}
        elif action == "create_circle":
            result = {"circle": circle_summary(places.create_circle(root, name=params.get("name"), text=words or "",
                                                                    model=params.get("model", ""), author=author))}
        elif action == "edit_circle":
            result = {"circle": circle_summary(places.edit_circle(
                root, params.get("circle_id"), author=author, expected_revision=params.get("expected_revision"),
                name=params.get("name"), text=words, model=params.get("model")))}
        elif action == "delete_circle":
            result = places.delete_circle(root, params.get("circle_id"), expected_revision=params.get("expected_revision"))
        else:
            saved = places.edit_place(root, place_id, action, author=author, **params)
            result = {"place": place_view(root, saved)} if saved else {}
        return json.dumps({"ok": True, **result}, ensure_ascii=False)
    except (places.PlaceError, ValueError, TypeError) as exc:
        if getattr(exc, "code", "") == "place_activation_owner_only":
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
                text="Only your human switches a place to Active, on its sheet or by a word in Main; propose it there."))
        blocked = getattr(exc, "code", "") == "place_owner_confirmation_required"
        return _publish_tool_result(ctx, ToolResult(status="blocked" if blocked else "error",
            code="RESOURCE_CONSTRAINT_BLOCKED" if blocked else "TOOL_ARG_ERROR",
            text=("⚠️ RESOURCE_CONSTRAINT_BLOCKED: " if blocked else "ERROR: ") + str(exc)))


def _cancel_presence_work(ctx: ToolContext, work_ref: str, reason: str = "") -> str:
    """Cancel work started from this presence binding (any of its conversations) or this turn's own tree."""

    from ouroboros.presence_authority import presence_caller_binding, presence_work_refusal
    from ouroboros.task_results import validate_task_id
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.join_ledger import _cancel_task

    try:
        task_id = validate_task_id(work_ref)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: PRESENCE_WORK_REF_INVALID: {exc}")))
    refusal = presence_work_refusal(ctx, task_id, drive_root=canonical_data_root(ctx), same_tree=True)
    # A speaker, or a root acting only for its binding: the binding authority decides, not speaker metadata.
    if refusal or presence_caller_binding(ctx) is None:
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "ERROR: PRESENCE_WORK_NOT_CORRELATED: " + (refusal.split(": ", 1)[-1] or "this is not a presence task."))))
    return _cancel_task(ctx, task_id, reason)


def get_tools() -> list[ToolEntry]:
    return [ToolEntry(name="configure_presence", schema={
        "name": "configure_presence",
        "description": "Inspect places and their connector choices; edit state, wakes, notes, folder or model. "
                       "Only your human switches a place to Active, on its sheet or by a word in Main; propose it there. "
                       "Only an owner-door turn may confirm an owner account. When your human changes a standing "
                       "instruction for a place, remove or replace the superseded note in this turn. Safety checks read "
                       "the current notes of places the host identifies for each call. A note that is their exact "
                       "words, written in an owner-door turn, is recorded as theirs; your summary is recorded as "
                       "yours and is no instruction of theirs. "
                       "Circles group places under shared words every place in one inherits beside its own notes: "
                       "create_circle/edit_circle (name, text, model)/delete_circle (its places stay connected and keep "
                       "their notes); set_circle puts place_id in circle_id (empty: none), at most one circle each; "
                       "set_description sets its short purpose line (empty clears), never a rule. Circle words "
                       "follow the note rule. Your human's words restrict every place in their circle but consent "
                       "only where they placed it. list shows each circle's words once; inspect shows the "
                       "words a place inherits and model_route {value, layer}; read them before you act in another place. "
                       "A place's empty model inherits its circle's, then consciousness; an empty circle model inherits consciousness. "
                       "A folder is an address, not a fence. "
                       "Use canonical account_id and conversation_id from the connector's tools, connection metadata or event route, "
                       "not display labels; an actor ID is not necessarily a conversation ID. "
                       "For connect, omitted or empty account_id leaves the account scope unrestricted; omitted or empty "
                       "conversation_id covers all conversations in that scope. For confirm_owner_account/revoke_owner_account, "
                       "omitted account_id inherits the place's account; empty targets only a confirmation with no account restriction. "
                       "Connect starts Silent using the installed connector's declaration; disconnect keeps room history.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "inspect", "connect", "disconnect", "set_state",
                "add_wake", "remove_wake", "add_note", "remove_note", "set_working_folder", "set_model",
                "confirm_owner_account", "revoke_owner_account", "set_circle", "set_description",
                "create_circle", "edit_circle", "delete_circle"]},
            "place_id": {"type": "string"}, "connector": {"type": "string"},
            "account_id": {"type": "string"}, "conversation_id": {"type": "string"},
            "state": {"type": "string", "enum": ["active", "silent", "paused"]},
            "fact": {"type": "string", "description": "A core wake fact or signal.<slug> declared by this live connector."},
            "text": {"type": "string"}, "note_id": {"type": "string"},
            "working_folder": {"type": "string"}, "model": {"type": "string"},
            "platform_actor_id": {"type": "string"}, "label": {"type": "string"},
            "circle_id": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"},
            "expected_revision": {"type": "integer", "description": "The place's revision; for a circle action, the circle's."},
        }, "required": ["action"]},
    }, handler=_configure_presence, timeout_sec=30),
        ToolEntry(
            name="presence_cancel_work",
            schema={
                "name": "presence_cancel_work",
                "description": (
                    "Request cancellation of independent work started from this "
                    "place, in this or another of its conversations, of the owner's work begun "
                    "by answering one of its questions, or of this turn's own children. The result "
                    "is a request receipt, not proof the work stopped; work of another place or "
                    "the owner's other tasks is refused."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "work_ref": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["work_ref"],
                },
            },
            handler=_cancel_presence_work,
            timeout_sec=15,
        ),
    ]
