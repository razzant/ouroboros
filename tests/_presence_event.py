"""A transport event payload for the Host's Presence turn tests."""

from ouroboros.presence_events import PresenceTurnEvent


def event_payload(event: PresenceTurnEvent) -> dict:
    return {"source_event_id": event.source_event_id, "provider": event.provider, "account_id": event.account_id,
            "conversation_id": event.conversation_id, "thread_id": event.thread_id,
            "conversation_key": event.conversation_key, "actor": dict(event.actor),
            "conversation": dict(event.conversation), "message": dict(event.message), "text": event.text}
