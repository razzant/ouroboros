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
    """Hold only the outward view; keep the durable answer, output identity and owed polling.

    Transports poll only deferred outcomes: a held reply's admitted work stays observed, and its own
    result meets the place state of its own poll.
    """
    from dataclasses import replace
    if (root is not None and result.outcome in {"message", "deferred"}
            and reply_hold(root, metadata, result.task_id, result.text)):
        return replace(result, outcome="deferred" if result.outcome == "deferred" and result.work_ref else "silent", text="")
    return result


def project_output(root, metadata, task_id, output):
    if (root is not None and output.get("outcome", "message") in {"message", "deferred"}
            and reply_hold(root, metadata, task_id, str(output.get("text") or ""))):
        return {**output, "outcome": "silent", "text": ""}
    return dict(output)


def _conversation_effect_tools(connector: str, effect_name: str) -> list:
    """The connector's declared tools of one effect whose targets all lie within one conversation identity (account,
    conversation, thread; a gesture's message too) and that name a conversation argument. Read from the published
    declarations the dispatcher uses; no connector is named here."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.tool_effects import EFFECT_GESTURE, effect_of

    if not connector:
        return []
    with registry._lock:
        tools = [(name, dict(tool)) for name, tool in registry._tools.items()
                 if isinstance(tool, dict) and tool.get("skill") == connector]
    found = []
    for name, tool in tools:
        effect = effect_of(name, ext_tool=tool)
        # A gesture acts on one message, so its door must name the argument carrying that message's id.
        gesture = effect.effect == EFFECT_GESTURE
        if (effect.declared and effect.effect == effect_name and effect.conversation
                and (effect.message or not gesture)
                and set(effect.targets) <= {"account_id", effect.conversation, effect.thread,
                                            *((effect.message,) if gesture else ())}):
            found.append((name, effect))
    return found


def conversation_send_tools(connector: str) -> list:
    """The declared ``effect: message`` doors through which words reach the place a focus speaks in."""
    from ouroboros.tool_effects import EFFECT_MESSAGE
    return _conversation_effect_tools(connector, EFFECT_MESSAGE)


def conversation_gesture_tools(connector: str) -> list:
    """Declared ``gesture`` doors: a nonverbal response (a reaction) on one message of a place, not speech."""
    from ouroboros.tool_effects import EFFECT_GESTURE
    return _conversation_effect_tools(connector, EFFECT_GESTURE)


def records_only(metadata) -> bool:
    """A place focus whose own connector speaks through such a tool: its finish is its record and people hear only
    what it sends (owner 06.11, 2A). A connector without one (a peer URL, mail recipients) keeps the final as its reply."""
    metadata = metadata if isinstance(metadata, dict) else {}
    presence = metadata.get("presence") or {}
    return (metadata.get("wake_cause") == "focus" and bool(reply_carrier(metadata))
            and bool(conversation_send_tools(str(presence.get("transport_skill") or ""))))


def records_only_now(ctx) -> bool:
    """``records_only`` for an Active place under Acts or Evolves. A Silent or Paused place and Advises keep today's
    held draft (the owner's rehearsal before activating a place); an unreadable place keeps it too."""
    metadata = getattr(ctx, "task_metadata", {}) or {}
    if not records_only(metadata):
        return False
    from ouroboros.consciousness_authority import is_observe_origin
    from ouroboros.tool_access import canonical_data_root
    if is_observe_origin(metadata):
        return False
    try:
        return get_place(canonical_data_root(ctx), reply_carrier(metadata)["place_id"]).state == "active"
    except Exception:  # a missing or unreadable place: the held path says so, as it does for every final
        return False


def riding_sends(ctx, calls) -> list:
    """This response's successful sends and gestures into the turn's own conversation: what reaches people beside the
    finish it rides with. Three proofs: a ``tool_delivered`` finish beside a call whose delivery receipts the host
    matched to this conversation (``presence_delivery_confirmed``, re-checked against ``confirmed_delivery_receipts``;
    the #2538 contract, kept; today only the frozen legacy completion fixture produces that outcome), the event's own
    connector's declared send or gesture tool addressing this conversation, or its gesture tool naming the waking
    message exactly (``_gesture_on_waking_message``). A refused or failed call is ``is_error`` and stays a result."""
    from ouroboros.place_guard import call_place_facts

    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    connector = str((metadata.get("presence") or {}).get("transport_skill") or "")
    gestures = dict(conversation_gesture_tools(connector))
    tools = {name for name, _effect in conversation_send_tools(connector)} | set(gestures)
    delivered = (getattr(ctx, "_presence_completion", None) or {}).get("outcome") == "tool_delivered"
    riding = []
    for call in calls or []:
        if not isinstance(call, dict) or call.get("is_error"):
            continue
        receipts = call.get("presence_delivery_confirmed") or (call.get("result_meta") or {}).get("presence_delivery_confirmed")
        if delivered and isinstance(receipts, (list, tuple, set)) and receipts:
            from ouroboros.presence_context import confirmed_delivery_receipts
            if {tuple(pair) for pair in receipts} <= confirmed_delivery_receipts(ctx):  # receipts the host matched to this conversation
                riding.append(call)
            continue
        if not carrier or call.get("tool") not in tools:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        facts = call_place_facts(ctx, str(call["tool"]), args)
        rows = facts.get("places") or []
        if (any(carrier["conversation_key"] in (row.get("conversation_keys") or ()) for row in rows)
                or (call["tool"] in gestures and _gesture_on_waking_message(metadata, gestures[call["tool"]], args)
                    and _waking_event_place(ctx, metadata, connector) == carrier["place_id"])):
            riding.append(call)
    return riding


def _gesture_on_waking_message(metadata, effect, args) -> bool:
    """A gesture door that need not name a thread lands in this turn's conversation only by its exact message:
    the waking message's id, conversation and (where targeted) account. Any other message stays a result to read."""
    event = (metadata.get("presence") or {}).get("event") or {}
    message_id = waking_message_id(event)
    return bool(message_id and str(args.get(effect.message) or "") == message_id
                and str(args.get(effect.conversation) or "") == str(event.get("conversation_id") or "")
                and ("account_id" not in effect.targets or "account_id" not in args
                     or str(args["account_id"]) == str(event.get("account_id") or "")))


def _waking_event_place(ctx, metadata, connector) -> str:
    """The current Place the waking message's retained account, conversation and thread resolve to, or ``""``.
    A threadless call naming that exact message resolves no row of a thread-scoped Place (a migrated legacy binding);
    the message's own thread does, through the same matcher dispatch uses."""
    from ouroboros.place_guard import _drive_root
    from ouroboros.places import PlaceError, place_for_target

    event = (metadata.get("presence") or {}).get("event") or {}
    root = _drive_root(ctx)
    targets = {key: [str(event[key])] for key in ("account_id", "conversation_id", "thread_id") if event.get(key)}
    if not root or not connector or not targets.get("conversation_id"):
        return ""
    try:
        place, _known = place_for_target(root, connector=connector, targets=targets, conversation="conversation_id",
                                         thread="thread_id", provider=str(event.get("provider") or ""))
    except PlaceError:
        return ""
    return place.place_id if place is not None else ""


def waking_message_id(event) -> str:
    """The waking message's provider id, only as the connector normalized it (``event.message.message_id``)."""
    message = event.get("message") if isinstance(event, dict) and isinstance(event.get("message"), dict) else {}
    value = message.get("message_id")
    return str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""


def speak_line(metadata) -> str:
    """How words and declared gestures reach this place: every declared door with its own arguments, ids prefilled
    from the event (the host picks none: a pick would be a verdict), and the finish's fate. A connector with a send
    door makes the finish a record; one with only gesture doors keeps posting a non-silent finish as words."""
    from ouroboros.consciousness_authority import is_observe_origin

    metadata = metadata if isinstance(metadata, dict) else {}
    presence = metadata.get("presence") or {}
    event, place = presence.get("event") or {}, presence.get("place") or {}
    label = place.get("label") or place.get("provider") or presence.get("place_id") or "this place"
    connector = str(presence.get("transport_skill") or "")
    focus = metadata.get("wake_cause") == "focus"
    tools = conversation_send_tools(connector) if focus else []
    gestures = conversation_gesture_tools(connector) if focus else []
    if not tools and not gestures:
        return f"Your non-silent finish is posted in {label} as your words."
    message_id = waking_message_id(event)

    def door(name, effect, kind):
        args = [f"{effect.conversation}={event.get('conversation_id') or '…'}"]
        if effect.thread and event.get("thread_id"):
            args.append(f"{effect.thread}={event['thread_id']}")
        if "account_id" in effect.targets and event.get("account_id"):
            args.append(f"account_id={event['account_id']}")
        if kind == "gesture":
            args.append(f"{effect.message}={message_id or '…'}")
        return f"{kind + ': ' if gestures else ''}{name} ({', '.join(args)})"

    doors = [door(name, effect, "words") for name, effect in tools] + [door(name, effect, "gesture") for name, effect in gestures]
    through = ("one of " if len(doors) > 1 else "") + "; ".join(doors)
    act = "send or gesture" if gestures else "send"
    state = str(place.get("state") or "")
    held = (f" ({label} is {state.title()}: a {act} is refused and your finish is held for your human, not posted)"
            if state in {"silent", "paused"} else
            f" (autonomy is Advises: a {act} is refused and your finish is held for your human, not posted)"
            if is_observe_origin(metadata) else "")
    target = ("" if not gestures else
              " A gesture is a response, not speech. Its prefilled message id is the waking message; another message's "
              "exact id from this conversation is valid too." if message_id else
              " A gesture is a response, not speech. This event carries no id for the waking message: name an exact "
              "message id you know from this conversation.")
    if not tools:
        return (f"Your non-silent finish is posted in {label} as your words{held}. A gesture reaches it only when you "
                f"call it, through {through}; to answer with the gesture alone, finish with silent=true.{target}")
    if not gestures:
        return (f"Words reach {label} only when you send them, through {through}; a send may ride in the same "
                f"response as your finish. Your finish is your record, never posted{held}.")
    return (f"Words or a gesture reach {label} only when you call its door, through {through}; either may ride in "
            f"the same response as your finish. Your finish is your record, never posted{held}.{target}")


def record_note(ctx) -> str:
    """The one return of a records-only finish that posted nothing and sent nothing in its response: the facts (where
    words go, what this turn already sent there, how to send) and the two choices. The mind decides; once per run."""
    from ouroboros.place_guard import this_turn_sends
    from ouroboros.presence_context import _source_conversation

    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    label, route = _source_conversation(metadata.get("presence") or {})
    sends = this_turn_sends(ctx, carrier["conversation_key"])
    pending = "; ".join(f"{row.get('tool')} at {row.get('at')}" + (f": {json.dumps(row['text'], ensure_ascii=False)}" if "text" in row else "")
                        for row in sends["calls_without_receipt"]) or "none"
    return (f"[FOCUS_FINISH_IS_A_RECORD] Nothing was posted: in {label} (route {route}) your finish is your record, and "
            f"people hear only what you send. Confirmed sent there by this turn: {sends['confirmed']}. Your own send calls "
            f"there without a receipt yet: {pending}. {speak_line(metadata)} Send it now, or finish again to keep this "
            "text as your record.")


def prepare_focus_reply(ctx, text: str, *, quiet: bool = False) -> tuple[bool, str]:
    """Speech follows the mind's selection, never delivery receipts: where the place has no in-conversation send
    tool an answer is new speech even after a tool sent an acknowledgement, while ``quiet`` (an explicit silent
    finish or a host-authored final) says nothing new. Where the place speaks by tool (``records_only_now``) the
    finish is the record: nothing is posted; a finish with words that sent nothing in its own response returns
    once with the facts (``record_note``), the next is accepted. An admitted handoff stays deferred either way;
    words that are not speech never ride it (``quiet`` in the completion)."""
    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    if not carrier:
        return True, ""
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.presence_continuation import _handoff_ref, selection_for
    records = records_only_now(ctx)
    if records and text and not quiet:
        from ouroboros.loop_messages import OWNER_REPLY_OWED_REFUSAL, owner_reply_owed
        if owner_reply_owed(ctx):  # R3 at the common exit: a records-only finish owes my human's reply as a silent one does
            return False, OWNER_REPLY_OWED_REFUSAL
    if (records and text and not quiet and getattr(ctx, "_completion_riding_send", None) is False
            and not getattr(ctx, "_records_only_noted", False)):
        ctx._records_only_noted = True  # once: the mind then sends, or finishes again to keep its record
        return False, record_note(ctx)
    if records:
        ctx._focus_record_only = True  # the loop's wake_silent usage follows the transport (R7)
    quiet = quiet or records
    root = canonical_data_root(ctx)
    held = ""
    if not quiet and text:
        subject = {"text": text, "destination": carrier["destination"], "place_id": carrier["place_id"]}
        digest = _reply_digest(carrier, text)
        held = reply_hold(root, metadata, ctx.task_id, text)
        if held:
            ctx._focus_reply_hold = held
        elif getattr(ctx, "_focus_checked_reply", None) != digest:
            from ouroboros.safety import check_safety
            allowed, message = check_safety("place_reply", subject, ctx=ctx)
            if not allowed:
                return False, message
            if message:  # a warning rides beside the words in the room's record, where the next round reads it
                record_notice(root, place_id=carrier["place_id"], conversation_key=carrier["conversation_key"],
                              task_id=ctx.task_id, text="Safety note on this turn's reply: " + message,
                              identity=f"{ctx.task_id}:{digest}:safety-note")
            ctx._focus_checked_reply = digest
    outcome = "deferred" if _handoff_ref(ctx) else "silent" if quiet or held else "message"
    old = getattr(ctx, "_presence_completion", {}) or {}
    selection = old.get("selection") if old.get("message") == text else None
    ctx._presence_completion = {"outcome": outcome, "message": text,
                                "selection": selection or selection_for(ctx, hashlib.sha256(text.encode()).hexdigest()),
                                **({"quiet": True} if quiet or held else {})}
    ctx._presence_completion_accepted = True
    return True, ""


def refusal_facts(ctx) -> str:
    """Host facts beside a refused automatic reply, for the round that follows: where a non-silent finish would be
    posted and to whom, what this turn already sent there (receipts and its own receipt-less calls), and that a
    silent finish is valid. Facts only; whether to post, rewrite or stay silent is the mind's decision."""
    from ouroboros.place_guard import this_turn_sends
    from ouroboros.presence_context import _source_conversation
    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    if not carrier or records_only_now(ctx):  # a records-only place posts no finish: the record note says so
        return ""
    label, route = _source_conversation(metadata.get("presence") or {})
    sends = this_turn_sends(ctx, carrier["conversation_key"])
    pending = "; ".join(f"{row.get('tool')} at {row.get('at')}" + (f": {json.dumps(row['text'], ensure_ascii=False)}" if "text" in row else "")
                        for row in sends["calls_without_receipt"]) or "none"
    return (f"\n[FOCUS_REPLY_REFUSED] This answer was not posted. A non-silent finish_task answer is posted as a new message "
            f"in {label} (route {route}) to the people there, in your own voice. Confirmed sent there by this turn: "
            f"{sends['confirmed']}. Your own send calls there without a receipt yet: {pending}. Silence is valid: "
            "finish_task(action=\"finish\", silent=true) keeps the answer as your internal note; summary reaches your human only.")


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
