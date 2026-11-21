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


def reply_addresses(metadata, anchor=None) -> tuple:
    """The selected input's further reply addresses as ``{"source_event_id", "reply_targets"}`` (its
    ``message.reply_targets``), and why they cannot be read (``""`` when they can). ``anchor`` is this turn's selected
    input (``presence_continuation.reply_anchor``); None reads the one kept at completion (``presence_reply_anchor``);
    none at all (``{}``, a promoted or legacy turn's form) is the waking event itself, while a given anchor without its
    ``source_event_id`` and ``message`` cannot be read and holds the reply, never falling back. A selected message
    naming none is the legacy boundary, ``(None, "")``: another message's set is never borrowed."""
    from ouroboros.presence_events import reply_targets
    if anchor is None:
        anchor = metadata.get("presence_reply_anchor")
    if anchor is None or anchor == {}:
        presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
        anchor = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    elif not isinstance(anchor, dict) or not isinstance(anchor.get("message"), dict) or not anchor.get("source_event_id"):
        return None, "the reply's selected anchor names no source_event_id and message"
    try:
        found = reply_targets(anchor.get("message") if isinstance(anchor.get("message"), dict) else {})
    except ValueError as exc:
        return None, str(exc)
    if found is None:
        return None, ""
    return {"source_event_id": str(anchor.get("source_event_id") or ""), "reply_targets": [dict(row) for row in found]}, ""


def reply_hold(root, metadata, task_id, text: str, *, anchor=None) -> str:
    """Current host facts only: never generate or buy another safety check on replay. The reply's further addresses
    (``reply_addresses`` of ``anchor``) are resolved now, as its dispatch would meet them."""
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
    from ouroboros.place_guard import reply_address_hold
    try:
        place = get_place(root, carrier["place_id"])
        state, label = place.state, place.label
    except PlaceError as exc:
        if exc.code != "place_not_found":
            raise
        place, state, label = None, "disconnected", carrier["place_id"]
    message = (f"Reply held: {label} is {state.title()}" if state in {"silent", "paused", "disconnected"}
               else "Reply held: autonomy is Advises" if is_observe_origin(metadata) else "")
    addresses, problem = reply_addresses(metadata, anchor)
    if not message and place is not None:
        message = (f"Reply held: the addresses it reaches cannot be read ({problem})" if problem
                   else reply_address_hold(root, place, carrier["destination"], addresses))
    if message:
        digest = _reply_digest(carrier, text, addresses)
        record_notice(root, place_id=carrier["place_id"], conversation_key=carrier["conversation_key"],
                      task_id=task_id, text=message, identity=f"{task_id}:{digest}:held", held_text=text)
    return message


def _reply_subject(carrier, text, addresses=None) -> dict:
    """The automatic reply as Safety and its digest see it: its words, retained destination and place, and, when its
    selected input names further addresses, that input's id and full set (``reply_addresses``)."""
    return {"text": text, "destination": carrier["destination"], "place_id": carrier["place_id"],
            **({"reply_anchor": addresses} if addresses is not None else {})}


def _reply_digest(carrier, text, addresses=None):
    subject = _reply_subject(carrier, text, addresses)
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


def _door_effect(name: str, tool, effect_name: str):
    """The effect that makes ``name`` a door of ``effect_name`` under the declaration ``tool`` binds, else None: its
    targets all lie within one conversation identity (account, conversation, thread; a gesture's message too) and it
    names a conversation argument. Its groups may write or read elsewhere, but words or a gesture of a group must land
    in that same identity."""
    from ouroboros.tool_effects import EFFECT_GESTURE, EFFECT_MESSAGE, effect_of

    effect = effect_of(name, ext_tool=tool)
    # A gesture acts on one message, so its door must name the argument carrying that message's id.
    gesture = effect.effect == EFFECT_GESTURE
    return effect if (effect.declared and effect.effect == effect_name and effect.conversation
                      and (effect.message or not gesture)
                      and set(effect.targets) <= {"account_id", effect.conversation, effect.thread,
                                                  *((effect.message,) if gesture else ())}
                      and all(group.effect not in (EFFECT_MESSAGE, EFFECT_GESTURE)
                              or (group.conversation == effect.conversation and group.thread in ("", effect.thread)
                                  and group.account in ("", "account_id")) for group in effect.touches)) else None


def _conversation_effect_tools(connector: str, effect_name: str) -> list:
    """The connector's declared doors of one effect (``_door_effect``), read from the published declarations the
    dispatcher uses now; no connector is named here."""
    from ouroboros import extension_registry_state as registry

    if not connector:
        return []
    with registry._lock:
        tools = [(name, dict(tool)) for name, tool in registry._tools.items()
                 if isinstance(tool, dict) and tool.get("skill") == connector]
    return [(name, effect) for name, tool in tools if (effect := _door_effect(name, tool, effect_name)) is not None]


def _ran_as(call, connector: str, effect_name: str, today: dict):
    """The door of ``effect_name`` a recorded call ran as: judged by the declaration it ran with
    (``tool_effects.retained_descriptor``); a row recorded before calls kept one reads ``today``'s doors."""
    from ouroboros.tool_effects import retained_descriptor

    name, ran_with = str(call.get("tool") or ""), retained_descriptor(call)
    if ran_with is None:
        return today.get(name)
    return _door_effect(name, ran_with, effect_name) if ran_with["skill"] == connector else None


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
    from ouroboros.tool_effects import EFFECT_GESTURE, EFFECT_MESSAGE, retained_descriptor

    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    connector = str((metadata.get("presence") or {}).get("transport_skill") or "")
    sends, gestures = dict(conversation_send_tools(connector)), dict(conversation_gesture_tools(connector))
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
        if not carrier:
            continue
        # Judged by the declaration the call ran with, so a later publication never reinterprets what it did.
        send, gesture = (_ran_as(call, connector, kind, today) for kind, today in ((EFFECT_MESSAGE, sends),
                                                                                    (EFFECT_GESTURE, gestures)))
        if send is None and gesture is None:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        facts = call_place_facts(ctx, str(call["tool"]), args, ext_tool=retained_descriptor(call))
        rows = facts.get("places") or []
        if (any(carrier["conversation_key"] in (row.get("conversation_keys") or ()) for row in rows)
                or (gesture is not None and _gesture_on_waking_message(metadata, gesture, args)
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
    from ouroboros.presence_continuation import _handoff_ref, reply_anchor, selection_for
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
        # The selected input this reply answers names its further addresses: the hold, Safety and the digest read
        # the same anchor and full set.
        anchor = reply_anchor(ctx, metadata)
        addresses, _problem = reply_addresses(metadata, anchor)
        subject = _reply_subject(carrier, text, addresses)
        digest = _reply_digest(carrier, text, addresses)
        held = reply_hold(root, metadata, ctx.task_id, text, anchor=anchor)
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
    posted and to whom (only where ``final_route`` reads a route; routeless or unsettled, it says that instead), what
    this turn already sent there (receipts and its own receipt-less calls), and that a silent finish is valid. Facts
    only; whether to post, rewrite or stay silent is the mind's decision."""
    from ouroboros.place_guard import this_turn_sends
    from ouroboros.presence_context import ROUTED, ROUTELESS, UNKNOWN, _source_conversation, final_route
    from ouroboros.tool_access import canonical_data_root
    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    if not carrier or records_only_now(ctx):  # a records-only place posts no finish: the record note says so
        return ""
    label, route = _source_conversation(metadata.get("presence") or {})
    try:
        state, why = final_route(canonical_data_root(ctx), metadata, getattr(ctx, "task_id", ""))
    except Exception:
        state, why = UNKNOWN, "its route could not be read now"
    posted = f"posted as a new message in {label} (route {route}) to the people there, in your own voice"
    fate = {ROUTED: f"A non-silent finish_task answer is {posted}.",
            ROUTELESS: f"No connector reads this run's final in {label} (route {route}): {why}. A non-silent finish_task "
                       "answer stays this work's record; only a connector tool reaches the people there.",
            }.get(state, f"Whether a connector reads this run's final is not settled ({why}); if one does, a non-silent "
                         f"finish_task answer is {posted}.")
    sends = this_turn_sends(ctx, carrier["conversation_key"])
    pending = "; ".join(f"{row.get('tool')} at {row.get('at')}" + (f": {json.dumps(row['text'], ensure_ascii=False)}" if "text" in row else "")
                        for row in sends["calls_without_receipt"]) or "none"
    return (f"\n[FOCUS_REPLY_REFUSED] This answer was not posted. {fate} Confirmed sent there by this turn: "
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
