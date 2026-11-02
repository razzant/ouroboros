"""``set_task_retention``: the owner's "keep this" or its release, for exactly one task (``owner_keep``).

A thin tool over the same writer as ``POST /api/tasks/{id}/owner-keep``. Only a root task calls it, and
only on an owner's words this turn holds: ``owner_source`` names one held owner message
(``owner_quote_handles``), and by default the turn's own originating owner message is used. The fact
carries that message's identity. A call with no held owner message writes nothing: a model's own
judgement is no owner approval, for a keep and for a release alike.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, publish_no_effect

_REASON_MAX = 500


def _refuse(ctx: ToolContext, code: str, text: str) -> str:
    return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", meta={"reason_code": code},
                                             text=f"{code}: {text} Nothing was saved."))


def _owner_row(ctx: ToolContext, handle: str) -> Tuple[str, Dict[str, Any], str]:
    """The held owner row this call stands on, by its handle or the turn's origin; why not, else ``""``."""
    from ouroboros.owner_relay import held_owner_messages

    held = held_owner_messages(ctx)
    if handle:
        if handle not in held:
            return "", {}, "OWNER_SOURCE_NOT_HELD: choose one of the held owner handles " + json.dumps(list(held))
        return handle, held[handle], ""
    origin = (getattr(ctx, "task_metadata", None) or {}).get("origin_message_ref") or {}
    for address, row in held.items():
        if (origin and str(row.get("client_message_id") or "") == str(origin.get("client_message_id") or "")
                and int(row.get("chat_id", 1)) == int(origin.get("chat_id", 1))):
            return address, row, ""
    return "", {}, ("OWNER_SOURCE_REQUIRED: this turn answers no owner message; name the owner's words with "
                    "owner_source (one of " + json.dumps(list(held)) + ")")


def _set_task_retention(ctx: ToolContext, task_id: str, keep: bool, reason: str, owner_source: str = "") -> str:
    from ouroboros.owner_keep import set_owner_keep
    from ouroboros.owner_relay import owner_text
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import resolve_ctx_lineage, validate_task_id
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.join_ledger import _is_delegated_task

    try:
        target = validate_task_id(str(task_id or "").strip())
    except ValueError as exc:
        return _refuse(ctx, "TASK_ID_INVALID", str(exc))
    if not isinstance(keep, bool) or not isinstance(reason, str) or not reason.strip():
        return _refuse(ctx, "ARGUMENT_INVALID", "keep must be true or false and reason a sentence.")
    if _is_delegated_task(ctx) or not resolve_ctx_lineage(ctx)["is_root_task"]:
        return _refuse(ctx, "ROOT_ONLY", "only a root task, where the owner speaks, may change what is kept.")
    address, row, error = _owner_row(ctx, str(owner_source or "").strip())
    if error:
        return _refuse(ctx, error.split(":", 1)[0], error.split(":", 1)[1].strip())
    source = {"basis": "owner_message", "address": address,
              "message_ref": build_owner_message_ref(chat_id=int(row.get("chat_id", 1)),
                                                     client_message_id=str(row.get("client_message_id") or ""),
                                                     ts=str(row.get("ts") or ""), text=owner_text(row)),
              "tool": "set_task_retention", "by_task_id": str(getattr(ctx, "task_id", "") or "")}
    result = set_owner_keep(canonical_data_root(ctx), target, keep=keep, source=source,
                            note=reason.strip()[:_REASON_MAX])
    if result["status"] in {"refused", "inflight"}:
        return publish_no_effect(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", meta={"reason_code": result["reason"]},
            text=f"TASK_RETENTION_{result['status'].upper()}: {result['reason']}. Nothing was saved. "
                 + json.dumps(result, ensure_ascii=False)))
    return json.dumps(result, ensure_ascii=False)


_DESCRIPTION = (
    "Keep, or stop keeping, the saved materials of exactly one task when the owner asks: its work folders, "
    "copies, environments, review copies, result pins and call journals stay until the owner releases them. "
    "It never stops running work or lossless compression, and never covers that task's children. "
    "It needs the owner's own words: by default the message this turn answers, or one of your held "
    "owner_quote_handles. The result says what is kept now, what had already gone (`retired`), and what "
    "could not be checked (`unknown`)."
)


def get_tools() -> List[ToolEntry]:
    return [ToolEntry("set_task_retention", {
        "name": "set_task_retention",
        "description": _DESCRIPTION,
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string", "description": "The one task whose materials to keep or release."},
            "keep": {"type": "boolean", "description": "true keeps them; false releases an earlier keep."},
            "reason": {"type": "string", "description": "One sentence: why, as the owner said it."},
            "owner_source": {"type": "string", "default": "",
                             "description": "A held owner_quote_handle; empty means the owner message this turn answers."},
        }, "required": ["task_id", "keep", "reason"], "additionalProperties": False},
    }, _set_task_retention)]
