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


# What this work's lineage may already have said in its place conversation, kept in the loop's trace (which every
# continuation serializes whole, a new execution ID included): ``{"sends": "none"}`` only from a positively fresh first
# execution (``seed_no_send``), ``"possible"`` from the first call that might speak, set before that call starts
# (``note_possible_sends``), and ``noted`` once the routeless reminder was shown. Absent is unknown, never none.
NO_SEND_KEY = "presence_no_send"
ROUTELESS_MARK = "[ROUTELESS_FINAL]"
# Calls whose words go to another task: what it does with them is that task's, so they may still reach the place.
_HANDED_TO_ANOTHER_TASK = frozenset({"tell_task", "forward_to_worker"})


def seed_no_send(ctx, trace) -> None:
    """A place-carrying root's positively fresh first execution has said nothing there yet: no earlier attempt,
    working recovery (``_working_recovery_lost``: a locator its loader could not read), park, timeout retry or
    Continue precedes it. Every other start leaves the fact absent: an unknown lineage is not a silent one."""
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(trace, dict) or not isinstance(metadata, dict) or not reply_carrier(metadata):
        return
    if (int(getattr(ctx, "task_attempt", None) or 1) > 1 or getattr(ctx, "_working_recovery_lost", False)
            or getattr(ctx, "working_recovery", None) is not None
            or getattr(ctx, "owner_wait_resume", None) or getattr(ctx, "budget_pause_resume", None)
            or metadata.get("continuation") or metadata.get("timeout_retry_from")):
        return
    if metadata.get("original_task_id") not in (None, "", getattr(ctx, "task_id", "")):
        return
    trace[NO_SEND_KEY] = {"sends": "none"}


def may_speak(ctx, call, *, ext_tool=None) -> bool:
    """Whether one model tool call might put words into the place conversation, read before it starts. Known not to:
    the completion controls and schema bookkeeping, the known pure reads, this mind's own memory, the Ouroboros-side
    delivery family (the local room, never the place: ``named_delivery_receipt``), receipt writes of its own record, a
    connector's declared read or gesture (an act, not words), and a declared message whose every attributable target is
    another conversation (``call_place_facts``). Every other call might, an opaque command or browser action included;
    a failed or timed-out call proves nothing either way."""
    from ouroboros.place_guard import call_place_facts
    from ouroboros.tool_capabilities import (
        COGNITIVE_MEMORY_TOOL_NAMES,
        COMPLETION_RECEIPT_TOOLS,
        OWNER_DELIVERY_TOOL_NAMES,
        PURE_READ_TOOLS,
        completion_control_call,
    )
    from ouroboros.tool_effects import EFFECT_GESTURE, EFFECT_MESSAGE, EFFECT_READ, effect_of
    from ouroboros.tools.tool_discovery import resolve_gateway_call

    function = resolve_gateway_call(call).get("function") or {}
    name = str(function.get("name") or "").strip()
    if (completion_control_call(name) or name in {"enable_tools", "list_available_tools"}
            or name in PURE_READ_TOOLS | COGNITIVE_MEMORY_TOOL_NAMES | OWNER_DELIVERY_TOOL_NAMES
            or name in COMPLETION_RECEIPT_TOOLS - _HANDED_TO_ANOTHER_TASK):
        return False
    raw = function.get("arguments")
    args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    effect = effect_of(name, ext_tool=ext_tool, ctx=ctx, args=args if isinstance(args, dict) else None)
    if effect.declared and effect.effect in (EFFECT_READ, EFFECT_GESTURE):
        return False
    if effect.effect != EFFECT_MESSAGE or not isinstance(args, dict):
        return True
    own = reply_carrier(getattr(ctx, "task_metadata", {}) or {}).get("conversation_key")
    facts = call_place_facts(ctx, name, args, ext_tool=ext_tool)
    rows = facts.get("places") or []
    return not (facts.get("attributable") and rows and all(own not in (row.get("conversation_keys") or ()) for row in rows))


def note_dispatch(ctx, trace, call, *, bound: bool = False) -> None:
    """Monotonically retain possible speech before submission, using the timeout-bound descriptor when present.
    This is observational: an unreadable declaration suppresses the optional note, never the tool call."""
    state = trace.get(NO_SEND_KEY) if isinstance(trace, dict) else None
    if not isinstance(state, dict) or state.get("sends") != "none":
        return
    try:
        ext_tool = None
        if bound:
            from ouroboros.tools.extension_dispatch import bound_dispatch_candidate
            name = str((call.get("function") or {}).get("name") or "").strip()
            ext_tool, _unavailable = bound_dispatch_candidate(ctx, name, consume=False)
        speaking = may_speak(ctx, call, ext_tool=ext_tool)
    except Exception:
        speaking = True
    if speaking:
        state["sends"] = "possible"


def note_possible_sends(ctx, trace, messages, *, interrupted: bool = False) -> None:
    """Before the last batch's first effect: the first of its unanswered calls that might speak (``may_speak``) turns
    a known silent lineage into a possibly speaking one. The ``pre_effect`` checkpoint carries it, so a call that times
    out before its handler runs, or a restart after any of them, never reads as none. An unreadable call might."""
    state = trace.get(NO_SEND_KEY) if isinstance(trace, dict) else None
    if not isinstance(state, dict) or state.get("sends") != "none":
        return
    from ouroboros.budget_pause import pending_tool_call_ids

    pending = set(pending_tool_call_ids(messages))
    if interrupted and pending:
        # The saved declaration may predate dispatch. Today's read declaration cannot exonerate an unanswered call.
        state["sends"] = "possible"
        return
    batch = next((row for row in reversed(messages) if isinstance(row, dict) and row.get("role") == "assistant"), {})
    for call in batch.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("id") not in pending:
            continue
        note_dispatch(ctx, trace, call)
        if state.get("sends") != "none":
            return


def routeless_note(ctx, limit_ctx, trace, text: str, candidate, origin: str = "prose") -> str:
    """The one host fact a voluntary wordy finish returns with when nobody would receive it (owner 03.11, option 3),
    or ``""``. Only for a final ``final_route`` reads ROUTELESS now, in an Active place under Acts or Evolves whose
    connector declares a send door, from a lineage known to have sent nothing there (``NO_SEND_KEY``), once, with room
    for one more round. Silence, empty words, a stop and every unsettled or legacy case get nothing. The mind decides;
    the host sends nothing."""
    state = trace.get(NO_SEND_KEY) if isinstance(trace, dict) else None
    if (not isinstance(state, dict) or state.get("sends") != "none" or state.get("noted") or not str(text or "").strip()
            or candidate is None or getattr(candidate, "quiet", False)):
        return ""
    try:
        return _routeless_note(ctx, limit_ctx, str(text), origin)
    except Exception:  # an optional hint: an unreadable fact shows nothing and the finish proceeds
        import logging
        logging.getLogger(__name__).warning("routeless final facts unavailable", exc_info=True)
        return ""


def note_rows(content: str, note: str, origin: str) -> list:
    """The rows the note's round adds: the final it answers (a selection is already its own row), then the note."""
    return ([] if origin == "selection" else [{"role": "assistant", "content": content}]) + [{"role": "user", "content": note}]


def _routeless_note(ctx, limit_ctx, text: str, origin: str) -> str:
    from ouroboros.consciousness_authority import is_observe_origin
    from ouroboros.place_guard import this_turn_sends
    from ouroboros.presence_context import ROUTELESS, _source_conversation, final_route
    from ouroboros.tool_access import canonical_data_root

    metadata = getattr(ctx, "task_metadata", {}) or {}
    carrier = reply_carrier(metadata)
    connector = str((metadata.get("presence") or {}).get("transport_skill") or "")
    doors = [name for name, _effect in conversation_send_tools(connector)] if carrier else []
    if not doors or is_observe_origin(metadata):
        return ""
    root = canonical_data_root(ctx)
    route, why = final_route(root, metadata, getattr(ctx, "task_id", ""))
    if route != ROUTELESS or get_place(root, carrier["place_id"]).state != "active":
        return ""
    label, source = _source_conversation(metadata["presence"])
    confirmed = this_turn_sends(ctx, carrier["conversation_key"])["confirmed"]
    # A gesture is no words, so it leaves this lineage silent, but the mind sees what it did beside the fact.
    gestures = dict(conversation_gesture_tools(connector))
    made = [f"{row['tool']} on message {(row.get('args') or {}).get(gestures[row['tool']].message, '?')}"
            + (" (failed)" if row.get("is_error") else "")
            for row in (getattr(limit_ctx, "llm_trace", None) or {}).get("tool_calls") or []
            if isinstance(row, dict) and row.get("tool") in gestures]
    note = (f"{ROUTELESS_MARK} Nobody receives this text: no connector reads this run's final in {label} ({why}), and "
            f"the host sends nothing for you. People there hear only what a connector tool sends: {', '.join(doors)} "
            f"(source route {source}). Confirmed sent there by this run: {confirmed}. Gestures you made there, which "
            f"are not words: {'; '.join(made) or 'none'}. Send it that way now, or finish again: finish_task with "
            "silent=true sends nothing, and a final with words stays only this work's record.")
    # The round is priced as the exact request it would send: this transcript with the answer and the note.
    probe = [*(getattr(limit_ctx, "messages", None) or []), *note_rows(text, note, origin)]
    return note if _round_affordable(ctx, limit_ctx, probe_messages=probe) else ""


def routeless_note_pending(trace, candidate, messages) -> bool:
    """A shown routeless note no model response has followed yet: its round exists only for the note. Read from the
    transcript and the candidate's own control, so a restored attempt reads the same fact."""
    from ouroboros.loop_delivery import ROUTELESS_NOTED_CONTROL

    state = trace.get(NO_SEND_KEY) if isinstance(trace, dict) else None
    if (not isinstance(state, dict) or not state.get("noted")
            or getattr(candidate, "finalization_control", "") != ROUTELESS_NOTED_CONTROL):
        return False
    shown = [index for index, row in enumerate(messages or []) if isinstance(row, dict) and row.get("role") == "user"
             and str(row.get("content") or "").startswith(ROUTELESS_MARK)]
    return bool(shown) and not any(isinstance(row, dict) and row.get("role") == "assistant"
                                   for row in messages[shown[-1] + 1:])


def _round_affordable(ctx, limit_ctx, *, ahead: bool = True, probe_messages=None) -> bool:
    """Whether a round meets none of the rails that would end the work there instead: its round limit (``ahead``: the
    round after this one; the loop checked the current one), a live execution control, the supervisor's or the task's
    own deadline (``local_deadline_remaining``) inside the finalization reserve, or a money stop
    (``_optional_round_stops``, for the request ``probe_messages`` would send, else the transcript now). A round the
    existing rails would turn into a forced wrap-up or a pause is not bought."""
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.loop_round_limits import local_deadline_remaining
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    rounds = getattr(limit_ctx, "max_rounds", None)
    waiter = getattr(ctx, "model_wait_context", None)
    reserve = effective_finalization_reserve_sec(ctx)
    window = dispatch_window_remaining_sec(deadline_ts=getattr(limit_ctx, "deadline_ts", None), reserve_sec=reserve)
    local = local_deadline_remaining(ctx)
    return not ((ahead and rounds is not None and int(limit_ctx.round_idx) >= rounds)
                or (waiter is not None and waiter.control_reason())
                or (window is not None and window <= 0) or (local is not None and local <= reserve)
                or _optional_round_stops(limit_ctx, getattr(ctx, "_cost_ceiling", None), probe_messages))


def _optional_round_stops(ctx, ceiling, probe_messages=None) -> bool:
    """Whether a round only an optional host hint asks for would meet a money stop now: authored-completion exhaustion,
    a soft-landing ceiling, a reached ceiling, or, under an explicit profile, a next request (``probe_messages``, its
    exact transcript) whose reservation, or room for it and one more, does not fit as ``_check_budget_limits`` prices
    it, or whose balances, price or second fit are unknown (an owner Resume's relaxed last fit excepted). Read on a
    copy of the usage: nothing is stamped, paused, finalized or reserved."""
    import copy
    import logging

    from ouroboros import task_pacing
    from ouroboros.loop_budget import (
        _ceiling_reached,
        _exact_reservations,
        _loop,
        _wrapup_global_remaining,
        authored_completion_budget_exhausted,
    )

    probe = copy.copy(ctx)
    probe.accumulated_usage = dict(ctx.accumulated_usage)
    if authored_completion_budget_exhausted(probe, None, ceiling):
        return True
    state = getattr(ceiling, "state", None)
    if state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND:
        return True
    if state != task_pacing.COST_CEILING_ACTIVE:
        return False
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    authority = task_pacing.COST_STOP_EXPLICIT if tool_ctx is None else task_pacing.cost_stop_authority(tool_ctx)
    tree = _loop()._loop_tree_accounting(refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC)
    deciding, _basis = task_pacing.resolve_deciding_spend(tree_cost_usd=(tree or {}).get("settled_usd"),
        task_cost_usd=ctx.accumulated_usage.get("cost"), root_cap_usd=ceiling.root_cap_usd)
    global_remaining = None if ctx.active_use_local else _wrapup_global_remaining()
    if authority == task_pacing.COST_STOP_EXPLICIT:
        if global_remaining is None and (ceiling.root_cap_usd is None or deciding is None):
            return True  # no balance the profile's reservations could be priced against: unknown, not funded
        balances = dict(root_cap_usd=ceiling.root_cap_usd, deciding_usd=deciding, global_remaining_usd=global_remaining)
        # Only the owner's Resume relaxation may leave the second fit undecided; its own stamp says it did.
        probe.accumulated_usage.pop("budget_resume_last_fit_admitted", None)
        try:
            fits, two_fit = _exact_reservations(
                probe, list(probe_messages if probe_messages is not None else ctx.messages), balances,
                int(ctx.accumulated_usage.get("_context_prompt_estimate") or 0),
                bool(getattr(tool_ctx, "_budget_resume_last_fit_relaxed", False)),
                _loop()._server_web_allowed_by_task(tool_ctx))
        except Exception:
            logging.getLogger(__name__).debug("The optional round could not be priced", exc_info=True)
            return True
        relaxed = "budget_resume_last_fit_admitted" in probe.accumulated_usage
        if fits is not True or two_fit is False or (two_fit is None and not relaxed):
            return True
    return _ceiling_reached(deciding, ceiling.ceiling_usd, authority)


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
