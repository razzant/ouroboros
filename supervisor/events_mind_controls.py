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
    from supervisor import queue
    from ouroboros.work_diagnostics import control_actor_reason

    def current_source(source_id):
        import copy
        with queue._queue_lock:
            meta = queue.RUNNING.get(source_id)
            if meta is None:
                task = next((t for t in queue.PENDING if t.get("id") == source_id), None)
                meta = {"task": task, "attempt": task.get("_attempt")} if task else None
            return copy.deepcopy(meta)
    if refusal := control_actor_reason(queue.DRIVE_ROOT, actor, task_id, current_source=current_source):
        return {"ok": False, "error": refusal}
    if action == "pause_task" and actor == task_id:
        return {"ok": False, "error": "self_pause_refused"}
    reason = str(evt.get("reason") or "")
    named = evt.get("accept_unknown") if action == "resume_task" and isinstance(evt.get("accept_unknown"), list) else ()
    with mind_control(actor, reason), accepting_unknown(named, reason):
        if action == "pause_task":
            # The owner reads this declared author and sentence; like ``mind_control``, it grants no power.
            return request_owner_pause(task_id, request_id=nonce,
                                       by={"actor": "ouroboros", "text": reason.strip(), "at": utc_now_iso()})
        if action == "resume_task":
            return resume_budget_paused_task(task_id)
        if action == "continue_task":
            return admit_continuation(task_id, action_nonce=nonce)
    return {"ok": False, "error": "unknown_mind_control"}


def handle_mind_task_control(evt: dict, ctx) -> None:
    from ouroboros.owner_mailbox import write_task_message

    try:
        outcome = _apply(evt)
    except Exception as exc:
        outcome = {"ok": False, "error": "control_outcome_unknown", "detail": str(exc)}
    record = {"ts": utc_now_iso(), "type": "mind_task_control_outcome",
              **{key: evt.get(key) for key in ("task_id", "action", "actor_task_id", "action_nonce", "reason")},
              **({"accept_unknown": evt["accept_unknown"]} if evt.get("accept_unknown") else {}),
              "actor": "mind", "outcome": outcome}
    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", record)
    write_task_message(ctx.DRIVE_ROOT, json.dumps(record, ensure_ascii=False), str(evt["actor_task_id"]),
                       msg_id="mind-control:" + str(evt["action_nonce"]), source_task_id="", provenance="system")
    ctx.bridge.push_log(record)
