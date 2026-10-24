"""Model-free attention decisions after the room has durably heard an event."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ouroboros import presence_admission
from ouroboros.places import FACT_LEGACY_ADDRESSED, PlaceError, attention_name_matches, get_place, wake_matches


def decide_attention(root: Path, admission, event) -> tuple[str, dict]:
    """Return silent / retry / focus, consulting current owner choices each time."""
    decision, info, _place = attention_snapshot(root, admission, event)
    return decision, info


def attention_snapshot(root: Path, admission, event):
    """Keep the exact Place used for this decision available to focus provenance."""
    try:
        place = get_place(root, admission.place.place_id)
    except PlaceError as exc:
        if exc.code != "place_not_found":
            raise
        return "silent", {"reason": "disconnected"}, None
    decision, info = _decide_for_place(root, place, event)
    return decision, info, place


def _decide_for_place(root, place, event):
    facts = event.facts if event.facts is not None else (FACT_LEGACY_ADDRESSED,)
    if event.actor.get("is_bot") is True:
        facts = (*facts, "from_bot")
    if place.state == "paused":
        return "silent", {"reason": "paused"}
    if not presence_admission.focus_enabled(root):
        return "silent", {"reason": "off"}
    if not wake_matches(place, facts) and not attention_name_matches(place, event.text, facts):
        return "silent", {"reason": "no_wake_match"}
    allowance = presence_admission.focus_allowance(root)
    if allowance.get("status") != "available":
        retry_after = 60  # transport backoff when there is no known allowance reset
        try:
            reset = datetime.fromisoformat(str(allowance.get("resets_at") or "").replace("Z", "+00:00"))
            retry_after = max(1, int((reset - datetime.now(timezone.utc)).total_seconds()))
        except (ValueError, TypeError):
            pass
        return "retry", {**allowance, "retry_after": retry_after}
    return "focus", allowance


def settle_heard(root: Path, task: dict, event, reason: str = "heard", *, captured_task_id: str | None = None):
    """Finalize on the accepted room row; heard-only arrivals create no task document."""
    from ouroboros.place_arrivals import record_heard
    return record_heard(root, task, event, reason, captured_task_id=captured_task_id)


def offer_to_author(root, task: dict, event) -> bool:
    """A current author hears a new event as context, without another focus generation."""
    from ouroboros.conversation_lease import accept_author_mail, live_author
    from ouroboros.owner_mailbox import write_task_message
    author = live_author(event.conversation_key, drive_root=root)
    if not author or author == task["id"]:
        return False
    import json
    text = json.dumps({"actor": dict(event.actor), "event": event.text,
                       "source_event_id": event.source_event_id, "conversation_key": event.conversation_key,
                       **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task}}, ensure_ascii=False)
    def append():
        if not write_task_message(root, text, author, source_task_id=task["id"], provenance="place_event",
                                  msg_id="place-event:" + task["metadata"]["presence_event_identity"],
                                  sender_origin={"place_id": task["metadata"]["focus"]["place_id"],
                                                 "conversation_key": event.conversation_key}):
            raise OSError("The current author could not retain this place arrival; retry.")
        return True
    return accept_author_mail(root, author, event.conversation_key, append)


def reopen_for_place_mail(ctx, controls) -> bool:
    """The final drain accepted context: let its author see it before ending.

    This does not revise owner authority or discard a reviewed answer. The mind
    can keep that answer after considering the arrival it accepted responsibility for.
    """
    if not controls.get("place_event"):
        return False
    agent = getattr(ctx, "owner_message_admission_agent", None)
    lock = getattr(ctx, "owner_message_admission_lock", None)
    if agent is None or lock is None:
        return False
    with lock:
        if not agent._busy or agent._current_task_id != ctx.task_id:
            return False
        agent._accepting_owner_messages = True
    return True
