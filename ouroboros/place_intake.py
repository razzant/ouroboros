"""Model-free attention decisions after the room has durably heard an event."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ouroboros import presence_admission
from ouroboros.dialogue_provenance import reaction_fact, reaction_words
from ouroboros.places import FACT_LEGACY_ADDRESSED, PlaceError, get_place, wake_matches

_REACTION_FIELDS = {"schema_version", "change", "old", "new", "counts", "target"}
_REACTION_LIST_LIMIT = 256
_REACTION_VALUE_LIMIT = 256
_REACTION_EXCERPT_LIMIT = 500


def _reaction_list_refusal(name: str, values, *, counted: bool = False) -> str:
    if not isinstance(values, list) or len(values) > _REACTION_LIST_LIMIT:
        return f"reaction.{name} must be a list of at most {_REACTION_LIST_LIMIT} provider values"
    for value in values:
        fields = {"type", "value", "count"} if counted else {"type", "value"}
        if (not isinstance(value, dict) or set(value) != fields or not isinstance(value["type"], str) or not value["type"]
                or not isinstance(value["value"], str)
                or max(len(value["type"]), len(value["value"])) > _REACTION_VALUE_LIMIT
                or (counted and (type(value["count"]) is not int or value["count"] < 0))):
            return f"reaction.{name} holds an invalid provider value"
    return ""


def _reaction_target_refusal(target, thread_id: str) -> str:
    if not isinstance(target, dict) or set(target) - {"message_id", "resolution", "author", "excerpt"}:
        return "reaction.target must name message_id and resolution"
    if not isinstance(target.get("message_id"), str) or not target["message_id"].strip():
        return "reaction.target.message_id must be the provider's exact id"
    if target.get("resolution") not in ("resolved", "unresolved"):
        return "reaction.target.resolution must be resolved or unresolved"
    if target["resolution"] == "unresolved" and thread_id:
        return "an unresolved reaction stays at conversation level: thread_id must be empty"
    if "author" in target:
        author = target["author"]
        if (not isinstance(author, dict) or set(author) - {"platform_actor_id", "display_name", "is_self"}
                or any(not isinstance(author[key], str) for key in author.keys() - {"is_self"})
                or ("is_self" in author and type(author["is_self"]) is not bool)):
            return "reaction.target.author holds invalid facts"
    if "excerpt" in target:
        excerpt = target["excerpt"]
        if (not isinstance(excerpt, dict) or set(excerpt) != {"text", "truncated"} or not isinstance(excerpt["text"], str)
                or len(excerpt["text"]) > _REACTION_EXCERPT_LIMIT or type(excerpt["truncated"]) is not bool):
            return f"reaction.target.excerpt must be text of at most {_REACTION_EXCERPT_LIMIT} characters and its truncated flag"
    return ""


def _reaction_actor_refusal(actor: dict, change) -> str:
    kind = actor.get("kind")
    if kind not in ("user", "chat", "anonymous"):
        return "a reaction's actor.kind must be user, chat or anonymous"
    if (kind == "anonymous") != (change == "count"):
        return "an aggregate count has an anonymous actor, and only it does"
    if kind == "anonymous":
        if "platform_actor_id" in actor or "id" in actor:
            return "an anonymous actor has no id"
    elif not isinstance(actor.get("platform_actor_id"), str) or not actor["platform_actor_id"].strip():
        return "a reaction's user or chat actor needs its platform_actor_id"
    if kind != "user" and (actor.get("authenticated") is not False or "is_bot" in actor):
        return "a chat or anonymous actor is authenticated=false and has no is_bot"
    return ""


def reaction_refusal(event: dict, staged_files=None) -> str:
    """Why this event's typed ``message.reaction`` is invalid, else ``""``. A ``reaction`` key that is no
    object or has no ``schema_version`` is not this fact, and that event keeps its ordinary path."""
    message = event.get("message")
    fact = message.get("reaction") if isinstance(message, dict) else None
    if not isinstance(fact, dict) or "schema_version" not in fact:
        return ""
    if not reaction_fact(message):
        return "reaction.schema_version must be 1"
    if set(fact) - _REACTION_FIELDS:
        return "reaction holds unknown fields"
    if event.get("text") or staged_files:
        return "a reaction carries no text or files"
    change, present = fact.get("change"), {key for key in ("old", "new", "counts") if key in fact}
    old, new = fact.get("old", []), fact.get("new", [])
    if change not in ("add", "remove", "replace", "count"):
        return "reaction.change must be add, remove, replace or count"
    if change == "count":
        if present != {"counts"}:
            return "a count carries only counts"
        refusal = _reaction_list_refusal("counts", fact["counts"], counted=True)
    else:
        if "counts" in present:
            return "only a count carries counts"
        refusal = _reaction_list_refusal("old", old) or _reaction_list_refusal("new", new)
        if not refusal and not {"add": not old and new, "remove": old and not new,
                                "replace": old and new and old != new}[change]:
            refusal = f"reaction.old and reaction.new do not describe {change}"
    if not refusal:
        refusal = (_reaction_actor_refusal(event["actor"], change)
                   or _reaction_target_refusal(fact.get("target"), str(event.get("thread_id") or "").strip()))
    if not refusal:
        try:
            json.dumps(fact, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            refusal = "reaction holds text that is not valid Unicode"
    return refusal


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
    reaction = reaction_fact(event.message)
    if reaction:
        # Context, whatever the wake list, owner or allowance says. A live author of the target's proven
        # conversation may hear it in the background; a bot only under the existing from_bot selection.
        bot = event.actor.get("is_bot") is True or "from_bot" in facts
        resolved = (reaction.get("target") or {}).get("resolution") == "resolved"
        heard = resolved and (not bot or bool(wake_matches(place, facts)))
        return "silent", {"reason": "reaction_context_only", "background": heard}
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
    disconnected place and a bot without its wake item stay heard-only. A reaction is never a wake."""
    if decision in {"focus", "retry"}:
        return True
    if details.get("reason") == "reaction_context_only":
        return False if details.get("background") else None
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
    reaction = reaction_fact(event.message)
    text = json.dumps({"actor": dict(event.actor), "event": event.text,
                       **({"non_speech": f"Reaction, not speech: {reaction_words(reaction)}"} if reaction else {}),
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
