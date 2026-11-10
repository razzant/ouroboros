"""Model-free attention decisions after the room has durably heard an event."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ouroboros import presence_admission
from ouroboros.dialogue_provenance import reaction_fact, reaction_words
from ouroboros.places import (FACT_FROM_BOT, PlaceError, get_place, observed_facts, own_account_event,
                              place_owner_accounts, wake_matches)

# What ``offer_arrival`` did: the reason its settlement records, or the mind's choice to make.
HANDED_TO_AUTHOR, HANDED_TO_WORK, ROUTE_TO_MIND = "current_author", "conversation_work", "route"
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
    facts = observed_facts(event)
    if place.state == "paused":
        return "silent", {"reason": "paused"}
    if not presence_admission.focus_enabled(root):
        return "silent", {"reason": "off"}
    if own_account_event(place, event, place_owner_accounts(root).get(place.place_id)):
        # My own post (through the connector or any tool acting as that account) is heard and kept, never a reason
        # to wake me or to mail my live author.
        return "silent", {"reason": "own_account"}
    reaction = reaction_fact(event.message)
    if reaction:
        # Context, whatever the wake list, owner or allowance says. A live author of the target's proven
        # conversation may hear it in the background; a bot only under the existing from_bot selection.
        bot = FACT_FROM_BOT in facts
        resolved = (reaction.get("target") or {}).get("resolution") == "resolved"
        heard = resolved and (not bot or bool(wake_matches(place, facts)))
        return "silent", {"reason": "reaction_context_only", "background": heard}
    routed = isinstance(admission.allowance, dict) and admission.allowance.get(ROUTE_TO_MIND) is True
    if not wake_matches(place, facts) and not routed:  # the Host's route travels by value to the runner's recheck
        return "silent", {"reason": "no_wake_match"}
    return allowance_attention(root)


def allowance_attention(root: Path) -> tuple[str, dict]:
    """A wake's own question: focus while the allowance is available, else retry at its reset."""
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


def settle_heard(root: Path, task: dict, event, reason: str = "heard", *, captured_task_id: str | None = None,
                 recipient: str = ""):
    """Finalize on the accepted room row; heard-only arrivals create no task document."""
    from ouroboros.place_arrivals import record_heard
    return record_heard(root, task, event, reason, captured_task_id=captured_task_id, recipient=recipient)


def _followup_in_scope(event) -> bool:
    """New words in a place's own ongoing exchange, from provider facts, never their meaning: a thread or topic,
    or a direct message. A threadless group's chatter is not addressed to its work, and an event with neither
    words nor a message fact (a deletion) says nothing new."""
    facts = event.facts or ()
    return ((bool(event.thread_id) or "direct_message" in facts)
            and (bool(str(event.text or "").strip()) or "any_message" in facts))


def author_hears(decision: str, details: dict, event) -> bool | None:
    """How a live author of this conversation may hear an arrival: True as a wake, False as
    background context, None not at all. Neither needs the wake allowance; Paused, Off, a
    disconnected place and a bot without its wake item stay heard-only. A reaction is never a wake;
    an untagged follow-up in its own thread, topic or direct message is, and other room chatter stays background."""
    if decision in {"focus", "retry"}:
        return True
    if details.get("reason") == "reaction_context_only":
        return False if details.get("background") else None
    bot = FACT_FROM_BOT in observed_facts(event)
    if decision != "silent" or details.get("reason") != "no_wake_match" or bot:
        return None
    return _followup_in_scope(event)


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


def _arrival_msg_id(task: dict) -> str:
    return "place-event:" + task["metadata"]["presence_event_identity"]


def _arrival_append(root, drive, recipient: str, task: dict, event, wake: bool, pin: dict, claim=None):
    """The one row of one arrival, for whichever recipient holds it: an ``append(admit=None)`` for a door.

    Its words carry the event's text, exact message and conversation facts, observed facts and
    attachments; where they leave a transport queue's events out, its ``context_source_ref`` reads
    them. ``wake``, the source event and its message also ride the entry as typed facts:
    the drain lets only a wake reopen a ready final (my human's own words are owner input either way) and
    anchors replies (``presence_continuation``).
    Inside the recipient's mail lock, admission precedes the first pin claim. A retry instead checks
    its existing pin. Once claimed, a holder is never released: the append may have landed even if
    its reply was lost and the holder has since consumed and retired its mailbox.
    """
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.place_arrivals import pin_recipient
    from ouroboros.owner_words import place_confirmed_words
    from ouroboros.places import owner_event_now
    source = _arrival_source(root, task, event) if isinstance(event.conversation.get("transport_queue"), dict) else None
    reaction = reaction_fact(event.message)
    place_id = task["metadata"]["focus"]["place_id"]
    # Provenance from the arrival's own author as the places confirm it now, never from the focus that hands it on
    # (a stranger's line handed on stays context) and never a verdict about whom it is for; the drain reads it again.
    confirmed_words = "" if reaction else place_confirmed_words(owner_event_now(root, place_id, event), event.text, event.message)
    text = json.dumps({"actor": dict(event.actor), "event": event.text,
                       **({"non_speech": f"Reaction, not speech: {reaction_words(reaction)}"} if reaction else {}),
                       "source_event_id": event.source_event_id, "conversation_key": event.conversation_key,
                       "message": dict(event.message), "conversation": arrival_conversation(event.conversation, source),
                       **({"context_source_ref": source} if source else {}),
                       **({"facts": list(event.facts)} if event.facts is not None else {}),
                       **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task}}, ensure_ascii=False)
    def append(admit=None):
        verdicts = []

        def inside_lock():
            verdicts.append(claim(admit) if claim is not None else
                            pin_recipient(root, task["id"], event.conversation_key, pin, expect=pin, admit=admit) == pin)
            return verdicts[-1]

        if write_task_message(drive, text, recipient, source_task_id=task["id"], provenance="place_event",
                              msg_id=_arrival_msg_id(task),
                              sender_origin={"place_id": place_id, "conversation_key": event.conversation_key},
                              place_arrival={"source_event_id": event.source_event_id,
                                             "message": dict(event.message), "wake": bool(wake),
                                             "account_id": event.account_id, "conversation_id": event.conversation_id,
                                             **({"confirmed_words": confirmed_words} if confirmed_words else {})},
                              admit=inside_lock):
            return True
        if verdicts == [False]:
            return False  # refused inside the mail lock before any write: ordinary admission takes the arrival
        # False also covers a torn append, an admission that could not be read and a lock never taken: effect unknown.
        raise OSError("The recipient could not retain this place arrival; retry.")
    return append


def arrival_owner_words(ctx, drive_root, entry: dict) -> str:
    """A drained place arrival's words as my human's, read when its reader takes it (``places.owner_event_now`` over
    the arrival's own author, on the canonical data root): a confirmation made after its offer counts, a withdrawn
    one no longer does. An entry written before those facts rode with it keeps the stamp its offer gave it."""
    arrival = entry.get("place_arrival") if isinstance(entry.get("place_arrival"), dict) else {}
    if "account_id" not in arrival:
        return str(arrival.get("confirmed_words") or arrival.get("owner_words") or "")
    try:
        words = json.loads(entry.get("text") or "{}")
    except ValueError:
        return ""
    message = arrival.get("message") if isinstance(arrival.get("message"), dict) else {}
    if not isinstance(words, dict) or reaction_fact(message):
        return ""
    from ouroboros.owner_words import place_confirmed_words
    from ouroboros.places import owner_event_now
    from ouroboros.tool_access import canonical_data_root
    root = canonical_data_root(ctx) if getattr(ctx, "drive_root", None) else drive_root
    facts = {"actor": words.get("actor") or {}, "account_id": arrival["account_id"],
             "conversation_id": arrival.get("conversation_id"), "message": message}
    place_id = str((entry.get("sender_origin") or {}).get("place_id") or "")
    return place_confirmed_words(owner_event_now(root, place_id, facts), words.get("event"), message)


def _author_pin(root, author: str) -> dict:
    return {"to": "author", "task_id": author, "drive": str(Path(root))}


def _work_pin(work: dict) -> dict:
    return {"to": "work", "task_id": work["task_id"], "drive": str(work["drive"])}


def _deliver(root, task: dict, event, pin: dict, prior: dict, accept, wake: bool = True) -> str:
    """Admit, claim and append in one mail transaction, after the local author's admission lock.

    A refusal before the claim leaves ``prior`` untouched and permits another offer. After the claim
    nothing releases the holder: even a missing mailbox can mean its reader already consumed it.
    Concurrent target selection is ordered by the existing pin CAS inside the mail lock. No late
    rollback can erase a retry's delivery, including one whose mailbox has already retired.
    """
    from ouroboros.place_arrivals import arrival_recipient, pin_recipient
    if arrival_recipient(root, task["id"]) != prior:
        return "pinned"

    def claim(admit=None):
        return (arrival_recipient(root, task["id"]) == prior
                and pin_recipient(root, task["id"], event.conversation_key, pin, expect=prior, admit=admit) == pin)

    drive, holder = Path(pin["drive"]), pin["task_id"]
    if accept(_arrival_append(root, drive, holder, task, event, wake, pin, claim)):
        return "held"
    return "pinned" if arrival_recipient(root, task["id"]) != prior else "refused"


def _retry_holder(root, task: dict, event, pin: dict, wake: bool) -> tuple[str, str] | None:
    """An arrival already given to its pinned holder: settled with it once held there; while missing there and that
    holder reads now (its drain has not ended, so its mailbox was never cleared), offered to it again, deduplicated
    under its lock. Otherwise the mind takes a waking one with that holder named and its effect unknown;
    no other work ever does. None: this offer's pin changed before its append; it wrote nothing and is offered afresh."""
    from ouroboros.conversation_lease import accept_author_mail, live_author
    from ouroboros.owner_mailbox import place_event_retained
    from ouroboros.place_arrivals import arrival_recipient
    from ouroboros.room_reader import conversation_works, lent_author_reads, work_reads, work_reads_now
    holder, key = str(pin.get("task_id") or ""), event.conversation_key
    reason = HANDED_TO_WORK if pin.get("to") == "work" else HANDED_TO_AUTHOR
    if place_event_retained(Path(pin["drive"]), holder, _arrival_msg_id(task)):
        return reason, holder
    append = _arrival_append(root, Path(pin["drive"]), holder, task, event, wake, pin)
    if pin.get("to") == "author":
        if live_author(key, drive_root=root) == holder and accept_author_mail(root, holder, key, append):
            return reason, holder
        if lent_author_reads(root, holder, key, _bindings(task)) and _accept_lent(root, holder, task, event)(append):
            return reason, holder
    else:
        work = next((row for row in conversation_works(root, key, _bindings(task)) if row["task_id"] == holder), None)
        if work is not None and work_reads(work, key) and append(admit=lambda: work_reads_now(work, key, _bindings(task))):
            return reason, holder
    if arrival_recipient(root, task["id"]) != pin:
        return None
    return (ROUTE_TO_MIND if wake else ""), holder


def _accept_lent(root, author: str, task: dict, event):
    from ouroboros.conversation_lease import accept_lent_author_mail
    from ouroboros.room_reader import lent_author_reads
    key = event.conversation_key
    return lambda append: accept_lent_author_mail(
        root, author, key, append, lambda: lent_author_reads(root, author, key, _bindings(task)))


def _offer_author(root, task: dict, event, wake: bool, pin: dict) -> tuple[str, str]:
    """``(outcome, author)``: the conversation's lease author, if one lives, and whether it now holds the arrival."""
    from ouroboros.conversation_lease import accept_author_mail, live_author
    author = live_author(event.conversation_key, drive_root=root)
    if not author or author == task["id"]:
        return "", ""
    accept = lambda append: accept_author_mail(root, author, event.conversation_key, append)  # noqa: E731
    return _deliver(root, task, event, _author_pin(root, author), pin, accept, wake), author


def offer_to_author(root, task: dict, event, *, wake: bool = True) -> bool:
    """A current author hears a new event as context, without another focus generation.

    A retry whose earlier append outlived its lost settlement finds the entry still held by this
    author, even after it closed its intake; an arrival given to another holder never reaches it.
    """
    from ouroboros.place_arrivals import arrival_recipient
    pin = arrival_recipient(root, task["id"])
    if pin.get("to") == "author":
        handed = _retry_holder(root, task, event, pin, wake)
        return offer_to_author(root, task, event, wake=wake) if handed is None else handed[0] == HANDED_TO_AUTHOR
    return not pin and _offer_author(root, task, event, wake, pin)[0] == "held"


def _bindings(task: dict) -> tuple:
    presence = task["metadata"]["presence"]
    return (presence.get("binding_id"), presence.get("place_id"))


def offer_arrival(root, task: dict, event, *, wake: bool) -> tuple[str, str]:
    """One source, one holder (``place_recipient`` on the arrival's own record, pinned before any append).

    The conversation's lease author first, whose focus can still tell its work; then its sole reader: a root, or an
    author parked for its review or private question, whose existing wait wakes and retakes the conversation before
    any model call. Several readers, or living work not reading this moment (preparing, past its final close), leave a
    waking arrival to the mind: an ordinary focus under the allowance, never plain heard. ``{to: mind}`` wrote to no
    work, so a later current author may still take it; a work or author holder never moves.
    ``(outcome, holder)``; outcome "" leaves ordinary admission."""
    from ouroboros.place_arrivals import arrival_recipient, pin_recipient
    from ouroboros.room_reader import (
        conversation_works, lent_author_controlled, lent_author_reads, lent_authors, work_not_reading_now, work_reads,
        work_reads_now,
    )
    key, pin = event.conversation_key, arrival_recipient(root, task["id"])
    if pin.get("to") in {"work", "author"}:
        return _retry_holder(root, task, event, pin, wake) or offer_arrival(root, task, event, wake=wake)
    done, author = _offer_author(root, task, event, wake, pin)
    if done == "held":
        return HANDED_TO_AUTHOR, author
    if done == "pinned" or not wake:
        return (offer_arrival(root, task, event, wake=wake) if done == "pinned" else ("", ""))
    if pin.get("to") == "mind":
        return ROUTE_TO_MIND, ""
    works = conversation_works(root, key, _bindings(task))
    readers = [work for work in works if work_reads(work, key)]
    named = lent_authors(root, key, exclude=task["id"])
    parked = [item for item in named if lent_author_reads(root, item, key, _bindings(task))]
    if len(readers) + len(parked) == 1:
        if parked:
            done = _deliver(root, task, event, _author_pin(root, parked[0]), {}, _accept_lent(root, parked[0], task, event))
            handed = HANDED_TO_AUTHOR, parked[0]
        else:
            done = _deliver(root, task, event, _work_pin(readers[0]), {},
                            lambda append: append(admit=lambda: work_reads_now(readers[0], key, _bindings(task))))
            handed = HANDED_TO_WORK, readers[0]["task_id"]
        if done == "held":
            return handed
        if done == "pinned":
            return offer_arrival(root, task, event, wake=wake)
        readers, parked = [], []
    if len(readers) + len(parked) > 1:
        why = "several_readers"
    elif (author or any(not lent_author_controlled(root, item) for item in named)
          or any(work_not_reading_now(work) for work in works)):
        why = "not_reading"
    else:
        return "", ""
    held = pin_recipient(root, task["id"], key, {"to": "mind", "why": why}, expect={})
    return (ROUTE_TO_MIND, "") if held.get("to") == "mind" else offer_arrival(root, task, event, wake=wake)


def _captured_arrival(root, metadata: dict, captured: str, identity: str, facts):
    """``(task, event)`` of one captured arrival rebuilt from its immutable room row, as the Host offered it
    (its provider thread whole: ``dialogue_store.whole_body``), its own author among its facts."""
    from ouroboros import dialogue_store
    from ouroboros.place_arrivals import _record
    from ouroboros.presence_runner import PresenceTurnEvent

    with dialogue_store.transaction(root, write=False) as conn:
        row = _record(conn, captured)
        body = dialogue_store.whole_body(conn, row["body"]) if row is not None else {}
    if not identity or body.get("presence_event_identity") != identity:
        return None
    transport = body.get("transport") or {}
    event = PresenceTurnEvent(
        source_event_id=str(body.get("client_message_id") or ""), provider=str(transport.get("provider") or ""),
        account_id=str(transport.get("account_id") or ""), conversation_id=str(transport.get("conversation_id") or ""),
        thread_id=str(transport.get("thread_id") or ""), conversation_key=str(transport.get("conversation_key") or ""),
        actor=dict(transport.get("actor") or {}), conversation=dict(transport.get("conversation") or {}),
        message=dict(transport.get("message") or {}), text=str(body.get("text") or ""),
        facts=None if facts == ["legacy_addressed"] or not isinstance(facts, list) else tuple(facts))
    task = {"id": captured, "metadata": {**metadata, "presence_event_identity": identity},
            **{key: body[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in body}}
    return task, event


def focus_arrival(root, metadata, selector, focus_task_id: str = ""):
    """``(task, event, wake)`` this place focus may hand on, else None. ``True``: the message that woke it, from
    host-bound metadata. A string: the ``source_event_id`` of a later message of the SAME conversation that the Host
    wrote into this focus's own mailbox. Never the latest text, a model's words, a quote or another room."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.presence_runner import presence_turn_task_id

    presence = metadata.get("presence") if isinstance(metadata, dict) else None
    stored = presence.get("event") if isinstance(presence, dict) else None
    if (not isinstance(stored, dict) or not isinstance(metadata.get("focus"), dict) or metadata.get("wake_cause") != "focus"
            or not metadata.get("presence_event_identity")):
        return None
    if selector is True:
        captured = presence_turn_task_id(str(presence.get("binding_id") or ""), str(stored.get("source_event_id") or ""))
        own = _captured_arrival(root, metadata, captured, metadata["presence_event_identity"], stored.get("facts"))
        return (*own, True) if own else None
    key = str(stored.get("conversation_key") or "")
    for row in drain_owner_entries(root, focus_task_id, set(), include_acknowledged=True):
        arrival = row.get("place_arrival") if isinstance(row.get("place_arrival"), dict) else {}
        if (row.get("provenance") != "place_event" or arrival.get("source_event_id") != selector
                or (row.get("sender_origin") or {}).get("conversation_key") != key):
            continue
        try:
            words = json.loads(row.get("text") or "{}")
        except ValueError:
            return None
        identity = str(row.get("msg_id") or "").removeprefix("place-event:")
        later = _captured_arrival(root, metadata, str(row.get("source_task_id") or ""), identity, words.get("facts"))
        if later and later[1].conversation_key == key and later[1].source_event_id == selector:
            return (*later, arrival.get("wake") is not False)
    return None


def hand_own_arrival(root, metadata, target: str, selector=True, focus_task_id: str = "") -> str:
    """The mind chose: one captured message this focus received goes to the one work it named, through the same door
    and row as a sole reader's. Its holder moves from the mind (or this focus) to that work once, under the arrival's
    own lock; a message any other holder has, or whose effect there is unknown, never moves. A refusal's reason,
    "" once that work holds it; a refusal writes nothing and resumes nothing."""
    from ouroboros.place_arrivals import arrival_recipient
    from ouroboros.room_reader import conversation_works, lent_author_reads, lent_authors, work_reads, work_reads_now
    found = focus_arrival(root, metadata, selector, focus_task_id)
    if found is None:
        return "arrival_not_a_place_focus" if selector is True else "arrival_not_delivered_here"
    task, event, wake = found
    pin, key = arrival_recipient(root, task["id"]), event.conversation_key
    if pin.get("to") in {"work", "author"} and pin.get("task_id") != focus_task_id:
        if pin.get("task_id") != target:
            return f"arrival_held_by:{pin.get('task_id')}"
        # Not held and not takeable now is no proof of absence: an earlier append's effect stays unknown.
        handed = _retry_holder(root, task, event, pin, wake)
        if handed is None:
            return hand_own_arrival(root, metadata, target, selector, focus_task_id)
        return "" if handed[0] in {HANDED_TO_WORK, HANDED_TO_AUTHOR} else f"arrival_effect_unknown:{target}"
    work = next((row for row in conversation_works(root, key, _bindings(task)) if row["task_id"] == target), None)
    if work is None and target in lent_authors(root, key, exclude=focus_task_id):
        if not lent_author_reads(root, target, key, _bindings(task)):
            return "arrival_target_not_reading"
        done = _deliver(root, task, event, _author_pin(root, target), pin, _accept_lent(root, target, task, event), wake)
        if done == "pinned":
            return hand_own_arrival(root, metadata, target, selector, focus_task_id)
        return "" if done == "held" else "arrival_target_not_reading"
    if work is None:
        return "arrival_target_not_listed"
    if not work_reads(work, key):
        return "arrival_target_not_reading"
    done = _deliver(root, task, event, _work_pin(work), pin,
                    lambda append: append(admit=lambda: work_reads_now(work, key, _bindings(task))), wake)
    if done == "pinned":
        return hand_own_arrival(root, metadata, target, selector, focus_task_id)
    return "" if done == "held" else "arrival_target_not_reading"


def conversation_work_section(root, presence: dict) -> str:
    """The focus's whole candidate set for its own message: every live root of this exact conversation, unpaged,
    and the holder its message already has, if any (that holder alone may take it; its effect may be unknown)."""
    from ouroboros.place_arrivals import arrival_recipient
    from ouroboros.presence_runner import presence_turn_task_id
    from ouroboros.room_reader import conversation_works, lent_author_reads, lent_authors, work_reads
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.recent_tasks import _preview
    event = presence.get("event") or {}
    key = str(event.get("conversation_key") or "")
    title = "## Work open in this conversation (host-authored facts)\n\n"
    try:
        works = conversation_works(root, key, (presence.get("binding_id"), presence.get("place_id")))
        states = [("reads new messages" if work_reads(work, key) else "takes no message now") for work in works]
        bindings = (presence.get("binding_id"), presence.get("place_id"))
        captured = presence_turn_task_id(str(presence.get("binding_id") or ""), str(event.get("source_event_id") or ""))
        parked = [(item, lent_author_reads(root, item, key, bindings),
                   ((load_task_result(root, item) or {}).get("metadata") or {}).get("presence") or {})
                  for item in lent_authors(root, key, exclude=captured)]
        pin = arrival_recipient(root, captured)
    except Exception as exc:  # building a turn must not fail on its listing: unknown is said, never shown as empty
        return title + f"Unavailable now ({type(exc).__name__}); recent_tasks reads current work."
    held = (f"This message was already given to {pin['task_id']}; whether it reached that work's model is unknown. "
            "Only that work may take it, never another.\n\n") if pin.get("to") in {"work", "author"} else ""
    if not works and not parked and not held:
        return ""
    lines = [f"- {work['task_id']} [{state}]: "
             + json.dumps(_preview(" ".join(str(work["task"].get("objective") or work["task"].get("description")
                                                or work["task"].get("text") or "").split()), 200), ensure_ascii=False)
             for work, state in zip(works, states)]
    lines += [f"- {item} [parked for its review or question; {'reads new messages' if reads else 'takes no message now'}]: "
              + json.dumps(_preview(" ".join(str(facts.get("observed_text") or "").split()), 200), ensure_ascii=False)
              for item, reads, facts in parked]
    return (title + held + "This list is complete. No work received this message unless you choose it: "
            "tell_task(task_id, arrival=true) hands it to one work that reads new messages, exactly as captured (your "
            "human's own words stay theirs); arrival=\"<source_event_id>\" hands a later message of this conversation that "
            "reached you. A note there stays your own. Queued, paused or held work takes none. "
            "get_task_result(task_id) reads a goal in full.\n\n" + "\n".join(lines))


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
        from ouroboros.conversation_lease import follow_loop_admission

        follow_loop_admission(ctx, True)  # its ready final reopens for the wake: so does a remote Host's view
    return True
