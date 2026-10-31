"""Presence event identity and origin validation for meeting calls (text turns parse in host_service_places)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


def presence_event_origin(payload: Any) -> dict[str, str]:
    """The provider facts a connected place is resolved by; empty when the payload lacks them."""
    source = payload if isinstance(payload, dict) else {}
    return {key: str(source.get(key) or "").strip() for key in ("provider", "account_id", "conversation_id", "thread_id")}


def presence_turn_event(payload: Any, admission: PresenceAdmission, *,
                        reporting_version: int = 0, continuing: int = 0) -> PresenceTurnEvent:
    """Validate a call's source event against its admitted place, as ``/presence/turn`` does.

    The transport's claimed conversation_key never chooses the host history
    or concurrency key; that key comes from the checked origin facts.
    """
    from ouroboros.places import WAKE_FACTS
    from ouroboros.presence_bindings import conversation_key

    expected = {"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                "conversation_key", "actor", "conversation", "message", "text"}
    if not isinstance(payload, dict) or set(payload) - {"facts"} != expected:
        raise PresenceTurnError("presence_event_invalid", "event")
    facts = payload.get("facts")
    if "facts" in payload and (not isinstance(facts, list) or any(fact not in WAKE_FACTS for fact in facts)):
        raise PresenceTurnError("presence_event_invalid", "event.facts")
    provider, account_id, conversation_id, thread_id = (
        str(payload.get(key) or "").strip()
        for key in ("provider", "account_id", "conversation_id", "thread_id"))
    origin = admission.origin
    if (provider != origin.transport or account_id != origin.account_id
            or (origin.conversation_id != "*" and conversation_id != origin.conversation_id)
            or (origin.thread_id and thread_id != origin.thread_id)):
        raise PresenceTurnError("presence_origin_mismatch", "event")
    event = PresenceTurnEvent(
        source_event_id=str(payload["source_event_id"] or "").strip(),
        provider=provider, account_id=account_id, conversation_id=conversation_id, thread_id=thread_id,
        conversation_key=conversation_key(provider, account_id, conversation_id, thread_id),
        actor=dict(payload["actor"]) if isinstance(payload["actor"], dict) else {},
        conversation=dict(payload["conversation"]) if isinstance(payload["conversation"], dict) else {},
        message=dict(payload["message"]) if isinstance(payload["message"], dict) else {},
        text=str(payload["text"] or ""), delivery_reporting_version=reporting_version,
        continuation_version=continuing, facts=tuple(facts) if facts is not None else None)
    if not event.source_event_id or not event.conversation_key or not event.actor:
        raise PresenceTurnError("presence_identity_missing", "event")
    if "transport_queue" in event.conversation:
        from ouroboros.presence_observations import validate_transport_queue
        validate_transport_queue(event.conversation["transport_queue"], event.conversation_key,
                                 event.source_event_id)
    return event
