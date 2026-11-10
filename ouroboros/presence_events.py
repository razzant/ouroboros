"""The one parser of a live Presence event: ``/presence/turn``, ``/call/open`` and the carrier's re-admission.

History imports keep their own narrower contract (``presence_history``)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ouroboros.dialogue_store import thread_references
from ouroboros.presence_admission import PresenceAdmission


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
    origin = admission.origin
    return (event.provider != origin.transport or event.account_id != origin.account_id
            or (origin.conversation_id != "*" and event.conversation_id != origin.conversation_id)
            or bool(origin.thread_id and event.thread_id != origin.thread_id))


_EVENT_KEYS = frozenset({"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                         "conversation_key", "actor", "conversation", "message", "text"})


def parse_presence_event(payload: Any, *, vouched_facts: Any = (), staged_files: Any = None,
                         reporting_version: int = 0, continuing: int = 0) -> PresenceTurnEvent:
    """A connector's live event, checked whole before any admission; malformed is ``PresenceEventInvalid``.

    ``vouched_facts``: the declared connector signals admissible beside ``WAKE_FACTS``. A door takes them from
    the authenticated connector's live publication (``places.declared_wake_signals``); a carrier re-admitting
    the Host's event takes the facts the Host admitted, by value, never its own process's registry.
    ``staged_files``: the files the event carries (a typed reaction carries none). The transport's claimed
    conversation_key never chooses the host history or concurrency key; it comes from the provider facts,
    which the admission then checks against the place (``outside_origin``).
    """
    from ouroboros.place_intake import reaction_refusal
    from ouroboros.places import WAKE_FACTS, validate_presence_identity
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
    facts, allowed = payload.get("facts"), set(WAKE_FACTS) | set(vouched_facts or ())
    if "facts" in payload and (not isinstance(facts, list)
                               or any(not isinstance(fact, str) or fact not in allowed for fact in facts)):
        raise PresenceEventInvalid("presence_payload_invalid", "event.facts",
                                   message="event.facts must be a list of declared wake facts")
    try:  # the provider's order, as history carries it
        source_order(payload["message"])
    except ValueError as exc:
        raise PresenceEventInvalid("presence_payload_invalid", "event.message.source_order", message=str(exc)) from None
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


def presence_turn_event(payload: Any, admission: PresenceAdmission, *,
                        reporting_version: int = 0, continuing: int = 0, vouched_facts: Any = ()) -> PresenceTurnEvent:
    """``parse_presence_event`` of a payload whose place is already admitted, then its origin checked."""
    event = parse_presence_event(payload, vouched_facts=vouched_facts, reporting_version=reporting_version,
                                 continuing=continuing)
    if outside_origin(admission, event):
        raise PresenceTurnError("presence_origin_mismatch", "event")
    return event
