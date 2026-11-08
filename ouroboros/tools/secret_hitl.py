"""``showSecretHitl(pluginName, fields)``: ask the owner for an integration secret in a protected form.

The name is camelCase on purpose: it is the public request name of the integration-secret
HITL contract (with the browser's ``saveSecretToken``), kept identical to that contract.

The tool carries no secret in either direction. It resolves the skill's reviewed
``secret_fields`` declaration (``ouroboros/secret_requests.py``), records one owner question
whose typed ``secret_request`` names the plugin, the field ids and their Settings keys, and
waits after this tool batch exactly as a required ``escalate`` question waits (same card
publication, same live continuation owner, same Stop and deadlines). The owner's values go
from the browser straight to Settings; the task wakes on ``secret_filled``, ``secret_replaced``
or ``secret_cancelled`` metadata delivered through the ordinary question mailbox.
"""

from __future__ import annotations

import copy
import pathlib
import uuid
from typing import Any, List

from ouroboros.contracts.skill_manifest import MAX_SECRET_FIELDS
from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.tool_catalog import ToolEntry
from ouroboros.tools.tool_context import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

TOOL_NAME = "showSecretHitl"


def _unavailable(ctx: ToolContext) -> str:
    """Why this caller cannot ask the owner for a secret now ("" when it can)."""
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.dialogue_provenance import presence_caller_binding

    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    if not str(getattr(ctx, "task_id", "") or "").strip():
        return "it requires an active task context"
    if presence_caller_binding(ctx) is not None:
        return "a Presence conversation does not ask the owner for secrets"
    if str(meta.get("parent_task_id") or "").strip() or str(meta.get("delegation_role") or "") not in ("", "root"):
        return "only a root task asks the owner; a delegated task reports the missing secret to its parent"
    if not callable(getattr(ctx, "owner_wait_callback", None)):
        return "it needs a root task with a live continuation owner to wait for the form"
    if is_a2a_chat_id(getattr(ctx, "current_chat_id", None)):
        return "this machine-to-machine chat has no owner form"
    return ""


def _show_secret_hitl(ctx: ToolContext, pluginName: str = "", fields: Any = None) -> str:  # noqa: N803 - public names
    from ouroboros.owner_quiz import record_asked
    from ouroboros.secret_requests import SecretRequestError, field_ids, question_text, resolve_secret_request
    from ouroboros.tools.core_artifacts import _quiz_host_facts
    from ouroboros.tools.owner_delivery import deliver_owner_event

    reason = _unavailable(ctx)
    if reason:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: showSecretHitl {reason}. No form was shown and no wait was selected."))
    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    root = pathlib.Path(str(meta.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    try:
        request = resolve_secret_request(root, pluginName, fields if isinstance(fields, list) else None)
    except SecretRequestError as exc:
        return argument_refusal(ctx, exc.code, [str(exc)], effect="No form was shown and no wait was selected.")
    task_id = str(ctx.task_id)
    quiz_id = uuid.uuid4().hex
    try:
        chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        chat_id = 0
    host_facts = _quiz_host_facts(ctx, root, task_id, chat_id)
    question = question_text(request)
    asked = record_asked(root, task_id, quiz_id=quiz_id, question=question, options=[], wait_for_answer=True,
                         chat_id=chat_id, host_facts=host_facts, secret_request=request)
    if asked.get("refused"):
        return argument_refusal(ctx, "SECRET_HITL_REFUSED", [
            "the owner-question address could not be registered" if asked["refused"] == "owner_questions_unavailable"
            else "this task already has the maximum number of open owner questions"],
            effect="No form was shown and no wait was selected.")
    mode = deliver_owner_event(ctx, {
        "type": "send_quiz", "chat_id": chat_id, "quiz_id": quiz_id, "question": question, "options": [],
        "stake": "", "assumption": "", "state": "open", "task_id": task_id, "host_facts": host_facts,
        "wait_for_answer": True,
    })
    ctx._owner_wait_requested = quiz_id
    ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "", 0
    names: List[str] = field_ids(request)
    return (f"OK: secret request {quiz_id} for {request['plugin_name']} ({', '.join(names)}) "
            f"{'accepted' if mode == 'live' else 'queued'} for delivery; the task waits after this tool batch "
            "while the owner fills the protected form. The values go to Settings and are not shown to you. "
            "You receive secret_filled or secret_replaced {pluginName, fields} after a confirmed save, or "
            "secret_cancelled. Any incoming mail or an owner hurry request also ends the wait; Stop and task "
            "deadlines still apply. If this task ends first, the form stays usable: a later save is recorded "
            "in the chat but does not resume this work.")


_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Ask the owner to enter an integration secret (a bot token, an API key a skill needs) in a protected "
        "in-chat form instead of the chat. pluginName is the installed skill; fields are ids its manifest "
        "declares in secret_fields (e.g. telegram: botToken). The value is saved to Settings and is not sent "
        "in this conversation; you then receive only secret_filled or secret_replaced {pluginName, fields}, "
        "or secret_cancelled. The owner sees whether a key is already set and may explicitly replace it. Root "
        "tasks only; the task waits after this tool batch. Saving grants and enables nothing; enabled skills may apply new values automatically: "
        "afterwards check the skill's grant and status. Never ask for a secret in chat instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pluginName": {"type": "string", "description": "Installed skill name, e.g. telegram."},
            "fields": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_SECRET_FIELDS,
                       "description": "Secret field ids declared by that skill's secret_fields, e.g. [\"botToken\"]."},
        },
        "required": ["pluginName", "fields"],
    },
}


def get_tools() -> List[ToolEntry]:
    # The form shows host-authored metadata only and its authority is the owner question's
    # (escalate is skip too): no value passes through this call to be checked.
    return [ToolEntry("showSecretHitl", copy.deepcopy(_SCHEMA), _show_secret_hitl, safety_policy="skip")]
