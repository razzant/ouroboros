"""The mind's explicit lifecycle choices, delegated to their existing owners."""

from __future__ import annotations

import json
import hashlib

from ouroboros.tools.registry import ToolEntry


def _continue_in_background(ctx, reason: str = "") -> str:
    from ouroboros.model_detach import arm_detach

    return json.dumps(arm_detach(ctx, reason), ensure_ascii=False)


def _request_control(ctx, action: str, task_id: str, reason: str, accept_unknown=None) -> str:
    from ouroboros.task_results import validate_task_id
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.tools.control_events import _emit_control_event
    from ouroboros.utils import utc_now_iso

    try:
        target = validate_task_id(task_id)
    except ValueError as exc:
        return json.dumps({"ok": False, "error": "invalid_task_id", "detail": str(exc)})
    from ouroboros.work_diagnostics import control_actor_reason
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    actor = str(getattr(ctx, "task_id", "") or "")
    refusal = control_actor_reason(getattr(ctx, "budget_drive_root", "") or ctx.drive_root, actor, target)
    if refusal:
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED",
            text=json.dumps({"ok": False, "error": refusal, "task_id": target})))
    if action == "pause_task" and target == actor:
        return json.dumps({"ok": False, "error": "self_pause_refused",
                           "note": "Use await_messages to sleep, or escalate the blocker to the owner."})
    call_id = current_tool_call_id()
    if not actor or not call_id:
        return json.dumps({"ok": False, "error": "control_call_identity_missing"})
    nonce = hashlib.sha256(json.dumps([actor, target, action, call_id]).encode("utf-8")).hexdigest()
    emitted = _emit_control_event(ctx, {
        "type": "mind_task_control", "action": action, "task_id": target,
        "actor": "mind", "actor_task_id": actor, "action_nonce": nonce,
        "reason": str(reason or ""), "ts": utc_now_iso(),
        **({"accept_unknown": [str(value) for value in accept_unknown if value]}
           if isinstance(accept_unknown, list) and accept_unknown else {})})
    if emitted == "serialization_failed":
        return json.dumps({"ok": False, "error": "control_event_serialization_failed"})
    return json.dumps({"ok": True, "status": "requested", "task_id": target, "action": action,
                       "action_nonce": nonce, "transport": emitted,
                       "note": "The supervisor decides through the owner's existing control. "
                               "The result will arrive in your task mailbox; this receipt is not completion."})


def _pause_task(ctx, task_id: str, reason: str = "") -> str:
    return _request_control(ctx, "pause_task", task_id, reason)


def _resume_task(ctx, task_id: str, reason: str = "", accept_unknown=None) -> str:
    return _request_control(ctx, "resume_task", task_id, reason, accept_unknown)


def _continue_task(ctx, task_id: str, reason: str = "") -> str:
    return _request_control(ctx, "continue_task", task_id, reason)


def _control_schema(name: str, description: str) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Exact id of host-listed work."},
                "reason": {"type": "string"}}, "required": ["task_id", "reason"]}}


def _accepting_schema(schema: dict) -> dict:
    schema["parameters"]["properties"]["accept_unknown"] = {
        "type": "array", "items": {"type": "string"},
        "description": "Ids a refusal listed in acceptable_unknown: held calls whose outcome is unknown "
                       "(their worker cannot be proven live or ended, or an outside effect is unconfirmed). "
                       "Naming them is your decision to go on without their results, recorded on the task with "
                       "your reason; they are never run again, and a call in a live worker is refused."}
    return schema


def mind_tool_entries() -> list:
    from ouroboros.model_detach import LOST_HANDLES

    return [ToolEntry("continue_in_background", {
        "name": "continue_in_background",
        "description": "Continue THIS work, with the same id and card, in a pooled worker. "
            "The handoff takes place after this tool batch completes; capacity, leases and holds may keep it queued. "
            "Live services, delegated runs, running children or retained/unreadable custody refuse the handoff. "
            + LOST_HANDLES + " From a pooled executor this is a no-op. Owner-question waits stay warm.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]},
    }, _continue_in_background),
        ToolEntry("pause_task", _control_schema("pause_task",
             "Request Pause of a listed work through the owner's tree-pause control. "
             "The owner reads your reason under the work's name: say plainly, in one short sentence without ids, "
             "why you paused it and whose decision it was. Started reviews may finish while the task pauses. "
             "Self-pause is refused: sleep with await_messages or escalate. The outcome arrives in your task mailbox."),
             _pause_task),
        ToolEntry("resume_task", _accepting_schema(_control_schema("resume_task",
             "Request Resume of listed paused work through the owner's existing grants. "
             "It preserves spent money, clocks and review history; capacity and holds still apply. "
             "Only the owner can release a Panic hold. The outcome arrives in your task mailbox.")), _resume_task),
        ToolEntry("continue_task", _control_schema("continue_task",
             "Request one successor episode of listed interrupted work through "
             "the same Continue action as the owner's button. Author-finished work, owner Stop/Panic and "
             "hard money/deadline stops refuse. Repeated delivery of this call reuses its one successor. "
             "The outcome arrives in your task mailbox."), _continue_task)]
