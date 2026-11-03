"""Inspect and edit places through the same writer as the owner Places sheet."""
from __future__ import annotations

import json
from typing import Any
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

# Frozen forced-finalization records still decode these wire outcomes.
PRESENCE_OUTCOMES = ("message", "silent", "tool_delivered", "deferred")


def _configure_presence(ctx: ToolContext, action: str, place_id: str = "", **params: Any) -> str:
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.places import AUTHOR_OWNER, AUTHOR_OUROBOROS, PlaceError, edit_place, get_place
    from ouroboros.place_projection import places_view, place_view, connect_declared_place
    from ouroboros.tool_access import canonical_data_root
    root = canonical_data_root(ctx)
    metadata = getattr(ctx, "task_metadata", {}) or {}
    owner = (bool(getattr(ctx, "is_direct_chat", False)) and run_origin({"metadata": metadata})["owner_ingress"]
             and metadata.get("wake_cause") != "focus" and metadata.get("initiator") != "consciousness")
    author = AUTHOR_OWNER if owner else AUTHOR_OUROBOROS
    try:
        if action == "list":
            result = places_view(root)
        elif action == "inspect":
            result = {"place": place_view(root, get_place(root, place_id))}
        elif action == "connect":
            result = {"place": place_view(root, connect_declared_place(root, **params))}
        else:
            saved = edit_place(root, place_id, action, author=author, **params)
            result = {"place": place_view(root, saved)} if saved else {}
        return json.dumps({"ok": True, **result}, ensure_ascii=False)
    except (PlaceError, ValueError, TypeError) as exc:
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
                       "Only an owner-door turn may confirm an owner account. A folder is an address, not a fence. "
                       "Connect starts Silent using the installed connector's declaration; disconnect keeps room history.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "inspect", "connect", "disconnect", "set_state",
                "add_wake", "remove_wake", "add_note", "remove_note", "set_working_folder", "set_model",
                "confirm_owner_account", "revoke_owner_account"]},
            "place_id": {"type": "string"}, "connector": {"type": "string"},
            "account_id": {"type": "string"}, "conversation_id": {"type": "string"},
            "state": {"type": "string", "enum": ["active", "silent", "paused"]},
            "fact": {"type": "string", "description": "A core wake fact or signal.<slug> declared by this live connector."},
            "text": {"type": "string"}, "note_id": {"type": "string"},
            "working_folder": {"type": "string"}, "model": {"type": "string"},
            "platform_actor_id": {"type": "string"}, "label": {"type": "string"},
            "expected_revision": {"type": "integer"},
        }, "required": ["action"]},
    }, handler=_configure_presence, timeout_sec=30),
        ToolEntry(
            name="presence_cancel_work",
            schema={
                "name": "presence_cancel_work",
                "description": (
                    "Request cancellation of independent work started from this "
                    "place, in this or another of its conversations (or of this turn's own "
                    "children). The result is a request receipt, not proof the work stopped; "
                    "work of another place or the owner's own tasks is refused."
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
