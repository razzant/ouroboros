"""Model-free attention decisions after the room has durably heard an event."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ouroboros import presence_admission
from ouroboros.places import FACT_LEGACY_ADDRESSED, PlaceError, get_place, wake_matches


def decide_attention(root: Path, admission, event) -> tuple[str, dict]:
    """Return silent / retry / focus, consulting current owner choices each time."""
    try:
        place = get_place(root, admission.place.place_id)
    except PlaceError as exc:
        if exc.code != "place_not_found":
            raise
        return "silent", {"reason": "disconnected"}
    facts = event.facts if event.facts is not None else (FACT_LEGACY_ADDRESSED,)
    if event.actor.get("is_bot") is True:
        facts = (*facts, "from_bot")
    if place.state == "paused":
        return "silent", {"reason": "paused"}
    if not presence_admission.focus_enabled(root):
        return "silent", {"reason": "off"}
    if not wake_matches(place, facts):
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


def author_hears(decision: str, details: dict, event) -> bool | None:
    """How a live author of this conversation may hear an arrival: True as a wake, False as
    background context, None not at all. Neither needs the wake allowance; Paused, Off, a
    disconnected place and a bot without its wake item stay heard-only."""
    if decision in {"focus", "retry"}:
        return True
    bot = event.actor.get("is_bot") is True or "from_bot" in (event.facts or ())
    return False if decision == "silent" and details.get("reason") == "no_wake_match" and not bot else None


def arrival_conversation(conversation, source=None) -> dict:
    """An arrival's exact conversation facts for its author. An attached transport-queue snapshot keeps
    every fact but its events there (under a burst each arrival would repeat the whole inbox) only with
    ``source``, the arrival's retained context source that returns them exactly; otherwise it stays whole."""
    facts = dict(conversation)
    queue = facts.get("transport_queue")
    if isinstance(queue, dict) and source:
        facts["transport_queue"] = {**{key: value for key, value in queue.items() if key != "events"},
                                    "events": "omitted here; memory_read(source_ref=<this arrival's "
                                              "context_source_ref>) returns them exactly, with every recorded fact"}
    return facts


def _arrival_source(root, task: dict, event) -> dict:
    """The arrival's exact recorded facts, retained once with the existing focus source (its room row's
    dedup record holds the pointer, so ``chat_history`` and ``memory_read`` rows show it too), as a
    ``memory_read(source_ref=...)`` locator."""
    from ouroboros.place_arrivals import intake_lock, retain_focus_source
    with intake_lock(root, event.conversation_key):
        ref = retain_focus_source(root, task["id"])
    return {key: value for key, value in ref.items() if key not in {"read", "canonical_root"}}


def offer_to_author(root, task: dict, event, *, wake: bool = True) -> bool:
    """A current author hears a new event as context, without another focus generation.

    Its words carry the event's text, exact message and conversation facts, observed facts and
    attachments; where they leave a transport queue's events out, its ``context_source_ref`` reads
    them. ``wake``, the source event and its message also ride the entry as typed facts:
    the drain lets only a wake reopen a ready final and anchors replies (``presence_continuation``).
    A retry whose earlier append outlived its lost settlement finds the entry still held by this
    author, even after it closed its intake.
    """
    from ouroboros.conversation_lease import accept_author_mail, live_author
    from ouroboros.owner_mailbox import place_event_retained, write_task_message
    author = live_author(event.conversation_key, drive_root=root)
    if not author or author == task["id"]:
        return False
    msg_id = "place-event:" + task["metadata"]["presence_event_identity"]
    source = _arrival_source(root, task, event) if isinstance(event.conversation.get("transport_queue"), dict) else None
    import json
    text = json.dumps({"actor": dict(event.actor), "event": event.text,
                       "source_event_id": event.source_event_id, "conversation_key": event.conversation_key,
                       "message": dict(event.message), "conversation": arrival_conversation(event.conversation, source),
                       **({"context_source_ref": source} if source else {}),
                       **({"facts": list(event.facts)} if event.facts is not None else {}),
                       **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task}}, ensure_ascii=False)
    def append():
        if not write_task_message(root, text, author, source_task_id=task["id"], provenance="place_event",
                                  msg_id=msg_id,
                                  sender_origin={"place_id": task["metadata"]["focus"]["place_id"],
                                                 "conversation_key": event.conversation_key},
                                  place_arrival={"source_event_id": event.source_event_id,
                                                 "message": dict(event.message), "wake": wake}):
            raise OSError("The current author could not retain this place arrival; retry.")
        return True
    return (accept_author_mail(root, author, event.conversation_key, append)
            or place_event_retained(root, author, msg_id))


def reopen_for_place_mail(ctx, controls) -> bool:
    """The final drain accepted context: let its author see it before ending.

    A wake arrival its model has not seen reopens, whether this drain or the round that
    only finalized delivered it; background alone keeps the ready final. This does not
    revise owner authority or discard a reviewed answer. The mind can keep that answer
    after considering the arrival it accepted responsibility for.
    """
    from ouroboros.presence_continuation import uncovered_wake
    if not controls.get("place_event") and not uncovered_wake(ctx):
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
