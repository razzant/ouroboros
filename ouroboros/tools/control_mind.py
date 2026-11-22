"""The mind's explicit lifecycle choices, delegated to their existing owners."""

from __future__ import annotations

import json
import hashlib

from ouroboros.tools.registry import ToolEntry


def _continue_in_background(ctx, reason: str = "") -> str:
    from ouroboros.model_detach import arm_detach

    return json.dumps(arm_detach(ctx, reason), ensure_ascii=False)


def _request_control(ctx, action: str, task_id, reason: str, accept_unknown=None, task_ids=None) -> str:
    """One control event per named work. ``task_ids`` (Pause and Resume): the roots, de-duplicated, each as an
    ordinary single event under its own nonce (as if named alone in this call), all marked with this call's id
    (``list_call``); a list Resume is a general one (``general``). Each mails its own outcome; nothing is kept
    between them."""
    from ouroboros.task_results import validate_task_id
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.tools.control_events import _emit_control_event
    from ouroboros.utils import utc_now_iso

    if task_ids is not None and not isinstance(task_ids, list):  # checked before the single or list form is chosen
        return json.dumps({"ok": False, "error": "invalid_task_ids", "detail": "task_ids must be a non-empty list."})
    if task_id and task_ids:
        return json.dumps({"ok": False, "error": "invalid_task_ids",
                           "detail": "Name one work in task_id or a non-empty list in task_ids, not both."})
    batch = bool(task_ids) and not task_id  # an empty list beside one id is no list
    if task_ids is not None and not task_id and not task_ids:
        return json.dumps({"ok": False, "error": "invalid_task_ids", "detail": "task_ids must be a non-empty list."})
    targets = []
    for value in task_ids if batch else [task_id]:
        try:  # the single form keeps the target's scalar contract; only list elements must be strings
            targets.append(validate_task_id((value if isinstance(value, str) else None) if batch else value))
        except ValueError as exc:  # one bad element refuses the whole call before anything is emitted
            return json.dumps({"ok": False, "error": "invalid_task_id", "detail": f"{value!r}: {exc}" if batch
                               else str(exc)})
    targets = list(dict.fromkeys(targets))
    actor = str(getattr(ctx, "task_id", "") or "")
    if action == "pause_task" and not batch and targets[0] == actor:
        return json.dumps({"ok": False, "error": "self_pause_refused",
                           "note": "Use await_messages to sleep, or escalate the blocker to the owner."})
    call_id = current_tool_call_id()
    if not actor or not call_id:
        return json.dumps({"ok": False, "error": "control_call_identity_missing"})
    rows = []
    for target in targets:
        if action == "pause_task" and target == actor:
            rows.append({"task_id": target, "ok": False, "error": "self_pause_refused"})
            continue
        nonce = hashlib.sha256(json.dumps([actor, target, action, call_id]).encode("utf-8")).hexdigest()
        emitted = _emit_control_event(ctx, {
            "type": "mind_task_control", "action": action, "task_id": target,
            "actor": "mind", "actor_task_id": actor, "action_nonce": nonce,
            "reason": str(reason or ""), "ts": utc_now_iso(),
            **({"list_call": call_id, **({"general": True} if action == "resume_task" else {})} if batch else {}),
            **({"accept_unknown": [str(value) for value in accept_unknown if value]}
               if isinstance(accept_unknown, list) and accept_unknown else {})})
        rows.append({"task_id": target, "action_nonce": nonce, "transport": emitted,
                     **({"ok": False, "error": "control_event_serialization_failed"}
                        if emitted == "serialization_failed" else {"ok": True, "status": "requested"})})
    if not batch:
        row = rows[0]
        if not row["ok"]:
            return json.dumps({"ok": False, "error": row["error"]})
        return json.dumps({"ok": True, "status": "requested", "task_id": row["task_id"], "action": action,
                           "action_nonce": row["action_nonce"], "transport": row["transport"],
                           "note": "The supervisor decides through the owner's existing control. "
                                   "The result will arrive in your task mailbox; this receipt is not completion."})
    return json.dumps({"ok": all(row["ok"] for row in rows), "status": "requested", "action": action,
                       "task_ids": targets, "list_call": call_id, "requests": rows,
                       "note": f"{len([row for row in rows if row['ok']])} requests sent, one per root. One outcome "
                               "per requested root will arrive in your task mailbox, each marked with this list_call; "
                               "count them and treat a root whose outcome did not arrive as unknown. This receipt is "
                               "not completion."}, ensure_ascii=False)


def _pause_task(ctx, task_id: str = "", reason: str = "", task_ids=None) -> str:
    return _request_control(ctx, "pause_task", task_id, reason, task_ids=task_ids)


def _resume_task(ctx, task_id: str = "", reason: str = "", accept_unknown=None, task_ids=None) -> str:
    return _request_control(ctx, "resume_task", task_id, reason, accept_unknown, task_ids)


def _continue_task(ctx, task_id: str, reason: str = "") -> str:
    return _request_control(ctx, "continue_task", task_id, reason)


def _control_schema(name: str, description: str) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Exact id of host-listed work."},
                "reason": {"type": "string"}}, "required": ["task_id", "reason"]}}


def _list_schema(schema: dict) -> dict:
    """Pause and Resume also take a list of roots (``task_ids``) in place of ``task_id``."""
    properties = schema["parameters"]["properties"]
    properties["task_id"]["description"] = "Exact id of one host-listed work; omit when task_ids is given."
    properties["task_ids"] = {
        "type": "array", "items": {"type": "string"},
        "description": "Exact ids of several host-listed ROOT works, in one call: each becomes its own ordinary "
                       "request, exactly as if named alone, and each mails its own outcome. A list is never "
                       "all-or-nothing and nothing waits for the whole: count the outcomes as they arrive, each "
                       "marked with this call's list_call, and treat a root whose outcome did not arrive as unknown. "
                       "An element that is not a valid task id refuses the whole call before anything is sent "
                       "(invalid_task_id names it; invalid_task_ids: both task_id and task_ids given, or an empty or "
                       "non-list task_ids). Consciousness and places are not works and are not listed. "
                       "Besides each door's own answers, a list row can refuse: sender_stopped (you, the sender, "
                       "were stopped after the call: nothing was done for this root, do not retry it); "
                       "sender_state_unreadable (whether you were stopped could not be read: nothing was done; "
                       "retry later or name the root alone); root_unverified or task_result_unreadable (the root's "
                       "own record could not be read or bound: nothing was done; retry later or look at it first); "
                       "not_a_root_task (a child: its root chooses its children, or name it alone if the owner's word "
                       "names it); owner_paused_by_name or pause_author_unknown (a Resume found a Pause Ouroboros "
                       "did not place: it stays; name that root alone by task_id when the owner's word names it)."}
    schema["parameters"]["required"] = ["reason"]
    return schema


def _accepting_schema(schema: dict) -> dict:
    schema["parameters"]["properties"]["accept_unknown"] = {
        "type": "array", "items": {"type": "string"},
        "description": "IDs from this Resume refusal's acceptable_unknown. Give your reason to resume the saved task. "
                       "Named local calls are never replayed. A delegated start may still work or spend money. "
                       "Its request becomes lookup-only. Live senders stay held."}
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
        ToolEntry("pause_task", _list_schema(_control_schema("pause_task",
             "Request Pause of a listed work through the owner's tree-pause control. "
             "The owner reads your reason under the work's name: say plainly, in one short sentence without ids, "
             "why you paused it, whose decision it was and what returns it: a time, an event, or only the word of "
             "whoever decided it. Later foci that list live work, consciousness included, see that sentence "
             "beside it; a press on work already paused keeps its first sentence. Started reviews may finish while "
             "the task pauses. "
             "Self-pause is refused: sleep with await_messages or escalate. To pause several roots at once (a "
             "general word to pause everything, say), name them in task_ids: one sentence for all, one outcome per "
             "root. A root already paused keeps its earlier Pause, author and sentence. "
             "The outcome arrives in your task mailbox.")), _pause_task),
        ToolEntry("resume_task", _accepting_schema(_list_schema(_control_schema("resume_task",
             "Request Resume of listed paused work through the owner's existing grants. "
             "A technical pause waits for the named billing or control record to become readable. "
             "The host rechecks the saved task's controls then. Resume a deliberate Pause when its named return "
             "has come or its author asks. A sleep wakes on what it chose. Your human's general word to "
             "continue returns what their general request paused; work they paused by name waits for their word "
             "about it or the return they named. "
             "It preserves spent money, clocks and review history; capacity and holds still apply. "
             "Only the owner can release a Panic hold. A list Resume (task_ids) is a general return: it lifts "
             "the Pauses Ouroboros itself placed and every other saved stop you may release, and keeps every Pause "
             "Ouroboros did not itself place, whoever is shown as its author now: the owner's "
             "(owner_paused_by_name) and one whose press declared no author (pause_author_unknown, often the "
             "owner's own from an older window). To lift such a Pause, name it alone by task_id when the owner's "
             "word names it. A child is its root's to choose (not_a_root_task). "
             "The outcome arrives in your task mailbox."))), _resume_task),
        ToolEntry("continue_task", _control_schema("continue_task",
             "Request one successor episode of listed work that was interrupted or that its author stopped "
             "as unfinished (finish_task action=stop), through the same Continue action as the owner's button. "
             "An ordinary finished answer, owner Stop/Panic and hard money/deadline stops refuse. A finite execution budget carries only its recorded unspent "
             "remainder; when that remainder cannot be read the request refuses as execution_budget_unknown "
             "and starts no successor, never with fresh time. An interrupted deep self-review saves no midpoint "
             "and refuses as deep_review_restart_required: request_deep_self_review starts a new one. "
             "Repeated delivery of this call reuses its one "
             "successor. The outcome arrives in your task mailbox."), _continue_task)]
