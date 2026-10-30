"""The automatic reply crosses the same current place state and Safety boundary as speech."""
from __future__ import annotations

import hashlib
import json
from ouroboros.places import PlaceError, conversation_chat_id, get_place, resolve_place


def record_notice(root, *, place_id: str, conversation_key: str, task_id: str, text: str, identity: str,
                  transport: dict | None = None, held_text: str | None = None):
    from ouroboros import dialogue_store
    from ouroboros.utils import utc_now_iso
    row = {"ts": utc_now_iso(), "direction": "system", "chat_id": conversation_chat_id(conversation_key),
           "user_id": 0, "text": text, "source": "presence", "place_id": place_id, "task_id": task_id,
           "type": "place_notice", "client_message_id": "place-notice:" + identity,
           "transport": transport or {"conversation_key": conversation_key},
           "presence_provenance": {"binding_id": place_id, "conversation_key": conversation_key}}
    if held_text is not None:
        # A state/autonomy hold precedes the Safety check; it is not a checked draft.
        row.update(held_text=held_text, held=True, checked=False)
    from ouroboros.presence_runner import _configured_gate
    gate = _configured_gate(root)
    key, lock = gate._conversation("notice:" + conversation_key)
    with lock, gate._file_lock(gate._conversation_file(key)):
        existing = dialogue_store.rows_by_client_id(root, row["client_message_id"], chat_id=row["chat_id"])
        if existing:
            return existing[0]
        return dialogue_store.append(root, "chat", row, feed=True)


def reply_carrier(metadata) -> dict:
    """Speech follows the retained destination, not a wake-only census stamp."""
    presence = metadata.get("presence") or {}
    event = presence.get("event") or {}
    place_id = presence.get("place_id") or (metadata.get("focus") or {}).get("place_id")
    if not place_id or not event.get("conversation_key"):
        return {}
    return {"place_id": place_id, "conversation_key": event["conversation_key"],
            "destination": event.get("destination") or {}}


def reply_hold(root, metadata, task_id, text: str) -> str:
    """Current host facts only: never generate or buy another safety check on replay."""
    carrier = reply_carrier(metadata)
    if not carrier and text:
        from pathlib import Path
        from ouroboros.places import load_places, places_path
        from ouroboros.presence_bindings import PRESENCE_BINDINGS_FILENAME
        presence = metadata.get("presence") or {}
        event = presence.get("event") or {}
        binding_id = presence.get("binding_id")
        # Pre-migration task bytes retain the old id. Resolve it through the
        # existing place store; isolated legacy readers have no place state.
        if (binding_id and event.get("conversation_key") and
                (places_path(root).exists() or (Path(root) / "state" / PRESENCE_BINDINGS_FILENAME).exists())):
            place = next((p for p in load_places(root) if binding_id in (p.place_id, *p.legacy_binding_ids)), None)
            carrier = {"place_id": place.place_id if place else binding_id,
                       "conversation_key": event["conversation_key"], "destination": event.get("destination") or {}}
    if not carrier or not text:
        return ""
    from ouroboros.consciousness_authority import is_observe_origin
    try:
        place = get_place(root, carrier["place_id"])
        state, label = place.state, place.label
    except PlaceError as exc:
        if exc.code != "place_not_found":
            raise
        state, label = "disconnected", carrier["place_id"]
    message = (f"Reply held: {label} is {state.title()}" if state in {"silent", "paused", "disconnected"}
               else "Reply held: autonomy is Advises" if is_observe_origin(metadata) else "")
    if message:
        digest = _reply_digest(carrier, text)
        record_notice(root, place_id=carrier["place_id"], conversation_key=carrier["conversation_key"],
                      task_id=task_id, text=message, identity=f"{task_id}:{digest}:held", held_text=text)
    return message


def _reply_digest(carrier, text):
    subject = {"text": text, "destination": carrier["destination"], "place_id": carrier["place_id"]}
    return hashlib.sha256(json.dumps(subject, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def project_reply(root, metadata, result):
    """Hold only the outward view; keep the durable answer and output identity."""
    from dataclasses import replace
    if (root is not None and result.outcome in {"message", "deferred"}
            and reply_hold(root, metadata, result.task_id, result.text)):
        return replace(result, outcome="silent", text="")
    return result


def project_output(root, metadata, task_id, output):
    if (root is not None and output.get("outcome", "message") in {"message", "deferred"}
            and reply_hold(root, metadata, task_id, str(output.get("text") or ""))):
        return {**output, "outcome": "silent", "text": ""}
    return dict(output)


def prepare_focus_reply(ctx, text: str, *, quiet: bool = False) -> tuple[bool, str]:
    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    if not carrier:
        return True, ""
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.presence_context import confirmed_delivery_receipts
    from ouroboros.presence_continuation import _handoff_ref, selection_for
    root = canonical_data_root(ctx)
    outcome = "silent" if quiet else "deferred" if _handoff_ref(ctx) else "tool_delivered" if confirmed_delivery_receipts(ctx) else "message"
    if outcome in {"message", "deferred"} and text:
        subject = {"text": text, "destination": carrier["destination"], "place_id": carrier["place_id"]}
        digest = _reply_digest(carrier, text)
        message = reply_hold(root, metadata, ctx.task_id, text)
        if message:
            ctx._focus_reply_hold = message
            outcome = "silent"
        elif getattr(ctx, "_focus_checked_reply", None) != digest:
            from ouroboros.safety import check_safety
            allowed, message = check_safety("place_reply", subject, ctx=ctx)
            if not allowed:
                return False, message
            ctx._focus_checked_reply = digest
    old = getattr(ctx, "_presence_completion", {}) or {}
    selection = old.get("selection") if old.get("message") == text else None
    ctx._presence_completion = {"outcome": outcome, "message": text,
                                "selection": selection or selection_for(ctx, hashlib.sha256(text.encode()).hexdigest())}
    ctx._presence_completion_accepted = True
    return True, ""


def record_late_delivery(root, skill: str, payload: dict, client_id: str):
    """Receipts are evidence even when the destination is now closed to writes."""
    if payload["state"] not in {"accepted", "delivered"}:
        return
    place = resolve_place(root, connector=skill, provider=payload["provider"], account_id=payload["account_id"],
                          conversation_id=payload["conversation_id"], thread_id=payload["thread_id"])
    if place is None:
        from ouroboros.places import places_of_connector
        place = next((p for p in places_of_connector(root, skill) if p.destination and
                      all(p.destination.get(key, "") in (payload[source], "*") for key, source in
                          (("transport", "provider"), ("account_id", "account_id"),
                           ("conversation_id", "conversation_id"), ("thread_id", "thread_id")))), None)
    if place is None or place.state not in {"silent", "paused"}:
        return
    from ouroboros.presence_bindings import conversation_key
    key = conversation_key(*(payload[k] for k in ("provider", "account_id", "conversation_id", "thread_id")))
    from ouroboros.dialogue_provenance import reaction_action_words, reaction_fact
    reaction = reaction_fact(payload.get("message"))
    action = reaction_action_words(reaction, payload["state"]) if reaction else ""
    words = "; ".join(part for part in (payload["text"], action) if part)  # the action follows any words sent beside it
    record_notice(root, place_id=place.place_id, conversation_key=key, task_id="",
                  text=f"Delivery reported while {place.label} is {place.state.title()}: {words}",
                  identity=client_id + ":late", transport={"conversation_key": key, "receipt": payload})
