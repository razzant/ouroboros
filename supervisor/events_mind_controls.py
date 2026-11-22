"""Mind lifecycle requests use the same owners as the owner's buttons."""

import json

from ouroboros.control_actor import accepting_unknown, mind_control
from ouroboros.utils import append_jsonl, utc_now_iso


def _apply(evt: dict) -> dict:
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import resume_budget_paused_task
    from supervisor.continuation_admission import admit_continuation

    task_id, actor = str(evt.get("task_id") or ""), str(evt.get("actor_task_id") or "")
    action, nonce = evt.get("action"), str(evt.get("action_nonce") or "")
    if not actor or not nonce:
        return {"ok": False, "error": "control_call_identity_missing"}
    if action == "pause_task" and actor == task_id:
        return {"ok": False, "error": "self_pause_refused"}
    if action == "switch_model":
        from ouroboros.owner_model import request, ModelActRefused

        if actor != task_id:
            return {"ok": False, "error": "model_actor_mismatch"}
        try:
            return request(task_id, {"request_id": nonce, "action": "set", "model": evt["model"],
                                     "use_local": bool(evt.get("use_local")), "cited": evt["cited"]},
                           source="owner_press" if evt.get("cited_source") == "owner_press" else "owner_words")
        except ModelActRefused as exc:
            return {"ok": False, "error": exc.code}
    reason = str(evt.get("reason") or "")
    named = evt.get("accept_unknown") if action == "resume_task" and isinstance(evt.get("accept_unknown"), list) else ()
    if evt.get("list_call") and (stopped := _sender_stopped(actor)):
        return stopped  # one row of a list call whose sender was stopped meanwhile: nothing is done for it
    with mind_control(actor, reason), accepting_unknown(named, reason):
        if action == "resume_task" and evt.get("general") and (kept := _kept_pause(task_id)):
            from ouroboros.work_stop import resume_outcome

            return resume_outcome(kept)
        if action == "pause_task":
            # The owner reads this declared author and sentence; like ``mind_control``, it grants no power.
            return request_owner_pause(task_id, request_id=nonce,
                                       by={"actor": "ouroboros", "text": reason.strip(), "at": utc_now_iso()})
        if action == "resume_task":
            return resume_budget_paused_task(task_id)
        if action == "continue_task":
            return admit_continuation(task_id, action_nonce=nonce)
    return {"ok": False, "error": "unknown_mind_control"}


def _sender_stopped(actor: str) -> dict | None:
    """A list call's rows are ordinary events emitted together; each first asks whether its sender was stopped
    since: its result cancelled, or a Stop pending on it or, for a timeout-retry attempt, on its LOGICAL root
    (a graceful cascade Stop stays keyed by that root), with the lineage validated as the owner's Pause does
    (``owner_pause.root_actor``). Then the row refuses and does nothing; an unreadable state refuses too."""
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.owner_pause import root_actor
    from ouroboros.task_results import load_task_result, resolve_task_lineage
    from supervisor import queue as q

    try:
        stored = load_task_result(q.DRIVE_ROOT, actor, strict=True) or {}
        lineage = resolve_task_lineage(actor, **{key: stored.get(key) for key in (
            "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
        members = {actor}
        if lineage["is_retry_root_attempt"]:
            if not root_actor(q.DRIVE_ROOT, lineage["root_task_id"], actor, strict=True):
                return {"ok": False, "error": "sender_state_unreadable"}
            members.add(lineage["root_task_id"])
        stopped = str(stored.get("status") or "") == "cancelled" or any(
            cancel_pending(q.DRIVE_ROOT, member, strict=True) for member in members)
    except Exception:
        return {"ok": False, "error": "sender_state_unreadable"}
    return {"ok": False, "error": "sender_stopped"} if stopped else None


def _kept_pause(task_id: str) -> dict | None:
    """The refusal a general (list) Resume owes ``task_id``, else ``None`` (the single door decides). Read where
    the single door and the owner's Pause read: the row in RUNNING or PENDING, its storage root (``_root_drive``),
    its result strictly, the fence of its tree (its logical root's for a retry attempt). In order: a stop the mind
    may not release (Panic first); a row whose root cannot be told, or a non-root (its root selects its children,
    owner Q9); a closed fence whose CREATING press was not Ouroboros's (``control_actor.creating_author``): the
    owner's (``owner_paused_by_name``) or one that declared nobody (``pause_author_unknown``, often the owner's own
    from before the window declared him). A later press filling in an author never makes it Ouroboros's."""
    from ouroboros.control_actor import control_actor, creating_author, observed_control_stop
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.task_results import load_task_result, resolve_task_lineage
    from ouroboros.work_stop import release_refusal
    from supervisor import queue as q
    from supervisor.owner_pause_control import _root_drive

    drive = _root_drive(q, task_id)
    with q._queue_lock:
        meta = q.RUNNING.get(task_id)
        row = ((meta.get("task") if isinstance(meta, dict) else None)
               or next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None))
    if row and (refusal := release_refusal(observed_control_stop(drive, row, budget=q.BUDGET_ROOT_FENCES),
                                           control_actor())):
        return refusal
    try:
        stored = load_task_result(drive, task_id, strict=True)
        if not isinstance(stored, dict):
            return {"ok": False, "error": "root_unverified"} if row else None  # unknown: task_not_pending
        lineage = resolve_task_lineage(task_id, **{key: stored.get(key) for key in (
            "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
        if not lineage["is_root_task"]:
            return {"ok": False, "error": "not_a_root_task", "root_task_id": lineage["root_task_id"]}
        fence = read_fence(drive, lineage["root_task_id"] if lineage["is_retry_root_attempt"] else task_id)
    except Exception:
        return {"ok": False, "error": "task_result_unreadable"}
    actor = creating_author(fence)["actor"] if fence_closed(fence) else "ouroboros"
    if actor != "ouroboros":
        return {"ok": False, "error": "owner_paused_by_name" if actor == "owner" else "pause_author_unknown",
                "actor": actor, "fence_id": str(fence.get("fence_id") or "")}
    return None


def handle_mind_task_control(evt: dict, ctx) -> None:
    from ouroboros.owner_mailbox import write_task_message

    try:
        outcome = _apply(evt)
    except Exception as exc:
        outcome = {"ok": False, "error": "control_outcome_unknown", "detail": str(exc)}
    record = {"ts": utc_now_iso(), "type": "mind_task_control_outcome",
              **{key: evt.get(key) for key in ("task_id", "action", "actor_task_id", "action_nonce", "reason")},
              **{key: evt[key] for key in ("list_call", "general") if evt.get(key)},  # a list call's row
              **({"accept_unknown": evt["accept_unknown"]} if evt.get("accept_unknown") else {}),
              "actor": "mind", "outcome": outcome}
    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", record)
    write_task_message(ctx.DRIVE_ROOT, json.dumps(record, ensure_ascii=False), str(evt["actor_task_id"]),
                       msg_id="mind-control:" + str(evt["action_nonce"]), source_task_id="", provenance="system")
    ctx.bridge.push_log(record)
