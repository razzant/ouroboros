"""The one parser of a live Presence event: ``/presence/turn``, ``/call/open`` and the carrier's re-admission.

History imports keep their own narrower contract (``presence_history``)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Mapping, Sequence

from ouroboros.dialogue_store import thread_references
from ouroboros.presence_admission import PresenceAdmission, origin_outside


class PresenceTurnError(ValueError):
    def __init__(
        self,
        code: str,
        field: str,
        *,
        attachment_manifest: Sequence[Mapping[str, Any]] = (),
        turn_ref: str = "",
        work_ref: str = "",
        disposition: str = "",
        delivery_reporting_version: int = 0,
    ) -> None:
        self.code = str(code or "presence_turn_failed")
        self.field = str(field or "presence_turn")
        self.turn_ref = str(turn_ref or "")
        self.work_ref = str(work_ref or "")
        self.disposition = disposition
        self.delivery_reporting_version = delivery_reporting_version
        self.attachment_manifest = [
            dict(row) for row in attachment_manifest if isinstance(row, Mapping)
        ]
        super().__init__(f"{self.code}: {self.field}")


class PresenceEventInvalid(PresenceTurnError):
    """A malformed connector event: every door refuses it 400 before the room hears it; ``message`` is the
    text door's long-standing words for it."""

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(code, field)
        self.message = message


def _invalid_facts():
    return PresenceEventInvalid("presence_payload_invalid", "event.facts",
                                "event.facts must be a list of declared wake facts")


@dataclass(frozen=True)
class PresenceTurnEvent:
    source_event_id: str
    provider: str
    account_id: str
    conversation_id: str
    thread_id: str
    conversation_key: str
    actor: Mapping[str, Any]
    conversation: Mapping[str, Any]
    message: Mapping[str, Any]
    text: str
    delivery_reporting_version: int = 0
    # Negotiated separately from receipts and never part of the event identity
    # (``presence_continuation``): 1 lets the transport take the initial envelope early.
    continuation_version: int = 0
    facts: tuple[str, ...] | None = None
    identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # The dialogue store keeps a provider thread by reference; an event rebuilt from its row reaches the model
        # only whole (``dialogue_store.whole_body``). One that still names a reference is refused, typed.
        if thread_references(self.conversation):
            raise PresenceTurnError("presence_event_unexpanded", "event.conversation")


def presence_event_origin(payload: Any) -> dict[str, str]:
    """The provider facts a connected place is resolved by; empty when the payload lacks them."""
    source = payload if isinstance(payload, dict) else {}
    return {key: str(source.get(key) or "").strip() for key in ("provider", "account_id", "conversation_id", "thread_id")}


def outside_origin(admission: PresenceAdmission, event: Any) -> bool:
    """Whether an event's provider facts lie outside the owner-created place/binding origin it was admitted by."""
    return origin_outside(admission.origin, event.provider, event.account_id, event.conversation_id, event.thread_id)


_EVENT_KEYS = frozenset({"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                         "conversation_key", "actor", "conversation", "message", "text"})
_REPLY_TARGET_KEYS = frozenset({"conversation", "effect", "match", "thread"})


def reply_targets(message: Any) -> tuple[dict[str, str], ...] | None:
    """``message.reply_targets``: the further literal addresses an automatic reply to this message reaches, as the
    connector knows them (``[]``: none); None when absent, a message from a connector that sends none (the legacy
    boundary). Each is ``{conversation, effect: "write", match: "scope"|"exact"|"exact_casefold", thread?}``: an id,
    never a selector, met in the reply's own route (the reply place's connector, its retained destination's provider
    and account, never the source event's). Anything else is ``ValueError``, never read as absent."""
    from ouroboros.places import MATCHES
    from ouroboros.tool_effects import EFFECT_WRITE

    if not isinstance(message, Mapping) or "reply_targets" not in message:
        return None
    raw, where = message["reply_targets"], "event.message.reply_targets"
    if not isinstance(raw, list):
        raise ValueError(f"{where} must be a list of addresses")
    found = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or not {"conversation", "effect", "match"} <= set(item) <= _REPLY_TARGET_KEYS:
            raise ValueError(f"{where}[{index}] holds conversation, effect and match, and optionally thread")
        if any(not isinstance(item.get(key, ""), str) for key in ("conversation", "thread")) or not item["conversation"].strip():
            raise ValueError(f"{where}[{index}]: conversation is a non-empty string id, thread a string")
        if item["effect"] != EFFECT_WRITE or item["match"] not in MATCHES:
            raise ValueError(f"{where}[{index}]: effect is {EFFECT_WRITE!r} and match is {' or '.join(MATCHES)}")
        thread = item.get("thread", "").strip()
        found.append({"conversation": item["conversation"].strip(), "effect": EFFECT_WRITE, "match": item["match"],
                      **({"thread": thread} if thread else {})})
    return tuple({json.dumps(entry, sort_keys=True): entry for entry in found}.values())


def parse_event_wire(payload: Any, *, staged_files: Any = None, reporting_version: int = 0,
                     continuing: int = 0) -> PresenceTurnEvent:
    """Validate the wire and derive its conversation key. A new intake also needs its connector's declaration."""
    from ouroboros.place_intake import reaction_refusal
    from ouroboros.places import WAKE_FACTS, is_wake_signal, validate_presence_identity
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_history import source_order

    if not isinstance(payload, dict) or set(payload) - {"facts", "identity"} != _EVENT_KEYS:
        raise PresenceEventInvalid("presence_event_invalid", "event", message="invalid presence event")
    try:  # the account the connector reads and sends as, when it negotiated ``presence_identity``
        identity = validate_presence_identity(payload["identity"]) if "identity" in payload else None
    except ValueError as exc:
        raise PresenceEventInvalid("presence_identity_invalid", "event.identity", message=str(exc)) from None
    actor = payload["actor"]
    if not isinstance(actor, dict) or ("authenticated" in actor and type(actor["authenticated"]) is not bool):
        raise PresenceEventInvalid("presence_actor_invalid", "event.actor",
                                   message="actor must be an object; authenticated must be a boolean")
    if refusal := reaction_refusal(payload, staged_files):
        raise PresenceEventInvalid("presence_reaction_invalid", "event.message.reaction", message=refusal)
    facts = payload.get("facts")
    if "facts" in payload and (not isinstance(facts, list) or any(
            not isinstance(fact, str) or (fact not in WAKE_FACTS and not is_wake_signal(fact)) for fact in facts)):
        raise _invalid_facts()
    try:  # the provider's order, as history carries it
        source_order(payload["message"])
    except ValueError as exc:
        raise PresenceEventInvalid("presence_payload_invalid", "event.message.source_order", message=str(exc)) from None
    try:  # an automatic reply's further addresses: absent is the legacy boundary, malformed never reads as absent
        reply_targets(payload["message"])
    except ValueError as exc:
        raise PresenceEventInvalid("presence_payload_invalid", "event.message.reply_targets", message=str(exc)) from None
    provider, account_id, conversation_id, thread_id = presence_event_origin(payload).values()
    event = PresenceTurnEvent(
        source_event_id=str(payload["source_event_id"] or "").strip(),
        provider=provider, account_id=account_id, conversation_id=conversation_id, thread_id=thread_id,
        conversation_key=conversation_key(provider, account_id, conversation_id, thread_id),
        actor=dict(actor),
        conversation=dict(payload["conversation"]) if isinstance(payload["conversation"], dict) else {},
        message=dict(payload["message"]) if isinstance(payload["message"], dict) else {},
        text=str(payload["text"] or ""), delivery_reporting_version=reporting_version,
        continuation_version=continuing, facts=tuple(facts) if facts is not None else None, identity=identity)
    if not event.source_event_id or not event.conversation_key or not event.actor:
        raise PresenceEventInvalid("presence_identity_missing", "event",
                                   message="presence event is missing identity facts")
    if "transport_queue" in event.conversation:
        from ouroboros.presence_observations import validate_transport_queue
        try:
            validate_transport_queue(event.conversation["transport_queue"], event.conversation_key,
                                     event.source_event_id)
        except ValueError as exc:
            raise PresenceEventInvalid("presence_payload_invalid", "event.conversation.transport_queue",
                                       message=str(exc)) from None
    return event


def _unpublished_facts(event, declared):
    from ouroboros.places import WAKE_FACTS

    return set(event.facts or ()).difference(WAKE_FACTS, declared)


def parse_presence_event(payload: Any, *, vouched_facts: Any = (), **wire) -> PresenceTurnEvent:
    """A carrier re-admits the event against the facts vouched for by its Host."""
    event = parse_event_wire(payload, **wire)
    if _unpublished_facts(event, vouched_facts or ()):
        raise _invalid_facts()
    return event


def connector_fact_labels(root, admission, event):
    """After place/origin admission, distinguish pending publication from an undeclared signal."""
    from ouroboros.extension_contributions import place_declarations_for_retry, place_fact_labels
    from ouroboros.extension_registry_state import live_contribution_projection
    from ouroboros.places import Place
    from ouroboros.places_migration import matching_declarations
    from ouroboros.presence_admission import WakeImportPending

    place = admission.place or Place(admission.binding_id, admission.transport_skill, admission.origin.transport)
    # Wire validity belongs to the connector's provider; a named contribution supplies this place's labels.
    origin = replace(place, contribution_id="") if place.contribution_id else place
    rows = matching_declarations(origin, live_contribution_projection())
    declared = place_fact_labels(rows)
    if _unpublished_facts(event, declared):
        expected = {fact["id"] for row in place_declarations_for_retry(root, origin)
                    for fact in row.get("facts", ()) if isinstance(fact, dict)}
        rows = matching_declarations(origin, live_contribution_projection())  # publication can overtake the lookup
        declared = place_fact_labels(rows)
        if missing := _unpublished_facts(event, declared):
            if missing <= expected:
                raise WakeImportPending()
            raise _invalid_facts()
    return place_fact_labels(matching_declarations(place, rows)) if place.contribution_id else declared


def presence_turn_event(payload: Any, admission: PresenceAdmission, *,
                        reporting_version: int = 0, continuing: int = 0, vouched_facts: Any = ()) -> PresenceTurnEvent:
    """``parse_presence_event`` of a payload whose place is already admitted, then its origin checked."""
    event = parse_presence_event(payload, vouched_facts=vouched_facts, reporting_version=reporting_version,
                                 continuing=continuing)
    if outside_origin(admission, event):
        raise PresenceTurnError("presence_origin_mismatch", "event")
    return event
