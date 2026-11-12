"""Dispatch-time refusals that follow from tool effects and place state.

The protected dispatch guard (``registry_guards._capability_resource_guard_result``)
asks ``dispatch_refusal`` once per call, the way it asks
``consciousness_authority.observe_argument_refusal``. Three rules live here:

* Advises (stored ``observe``): an extension or MCP call whose effect is not a
  declared read is refused, declared or not.
* Silent / Paused: every non-read effect aimed at such a place is refused
  from every focus (Main, tasks, schedules, wakes). A connector write whose
  target cannot be attributed is refused while any of that connector's places
  is Silent or Paused, and the reason says so. An explicit account outside all
  applicable place scopes is attributable without becoming a place.
* One author per conversation: a send into a conversation another turn is
  authoring is refused with that author's task id (hand the intent over with
  ``tell_task``).

Place lookup and lease ownership belong to their stores, not this policy.
"""

from __future__ import annotations

from typing import Any, Mapping

from ouroboros import places, conversation_lease
from ouroboros.tool_effects import (EFFECT_MESSAGE, EFFECT_READ, ToolEffect, argument_value,
                                    bound_declaration, connector_of, effect_of, target_values)

_PREFIX = "⚠️ RESOURCE_CONSTRAINT_BLOCKED: "
PLACE_BINDING_VERSION = 1
_DELIVERY_TARGET_FIELDS = ("provider", "account_id", "conversation_id", "thread_id")


def _metadata(ctx: Any) -> Mapping[str, Any]:
    value = getattr(ctx, "task_metadata", {})
    return value if isinstance(value, Mapping) else {}


def _drive_root(ctx: Any) -> Any:
    return (_metadata(ctx).get("budget_drive_root") or getattr(ctx, "budget_drive_root", "")
            or getattr(ctx, "drive_root", None))


def _counted_accounts(ctx: Any) -> Mapping[str, Any]:
    """The owner accounts each place counts now (``places.place_owner_accounts``): a sending identity reads as my
    human's wherever their bot or app confirms them; {} when the places cannot be read (each place's own rows)."""
    try:
        return places.place_owner_accounts(_drive_root(ctx)) if _drive_root(ctx) else {}
    except places.PlaceError:
        return {}


class _PlacesUnreadable(RuntimeError):
    """The place store needs repair; a connector's writes cannot be attributed."""


def _declares_place(connector: str, ext_tool: Any = None) -> bool:
    """Whether this skill's publication declares a place (it is a connector): the one a bound call was read with."""
    from ouroboros.extension_registry_state import live_contribution_projection

    if (bound := bound_declaration(ext_tool)) is not None:
        return bool(bound.get("provides_place"))
    return any(row.get("provides") == "place" and row.get("skill") == connector
               for row in live_contribution_projection())


def _call_destinations(ctx: Any, name: str, args: Mapping[str, Any], ext_tool: Any, is_mcp: bool):
    """Resolve each recipient separately so an Active To never hides a Silent Cc. A call its connector declares as
    one that delivers words (``sends.when``) is a ``message`` here, routed by that declaration's selectors."""
    effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx, args=args)
    targets = target_values(effect, args)
    connector, root = connector_of(name, ext_tool), _drive_root(ctx)
    if not connector or root is None:
        return effect, targets, [], [], True
    try:
        siblings = places.places_of_connector(root, connector)
    except places.PlaceError as exc:
        # An unreadable place store never blocks skills that connect no place; a
        # connector's writes wait for the repair (``dispatch_refusal`` names it).
        if not _declares_place(connector, ext_tool):
            return effect, targets, [], [], True
        raise _PlacesUnreadable(str(exc)) from exc
    routing = {key: targets[key] for key in ("account_id", effect.conversation, effect.thread) if key in targets}
    destinations = [{**routing, key: [value]} for key in (effect.targets or (effect.conversation,))
                    for value in targets.get(key, [])]
    resolved, attributable = [], bool(destinations)
    for destination in destinations or [targets]:
        place, known = places.place_for_target(root, connector=connector, targets=destination,
                                             conversation=effect.conversation, thread=effect.thread)
        attributable = attributable and known
        if place is not None and place not in resolved:
            resolved.append(place)
    return effect, targets, resolved, siblings, attributable


def _conversation_keys(place: Any, effect: Any, targets: Mapping[str, list[str]], *, account_id: str = "") -> list[str]:
    from ouroboros.presence_bindings import conversation_key

    conversations = targets.get(effect.conversation, []) or ([place.conversation_id] if place.conversation_id else [])
    threads = targets.get(effect.thread, []) or [""]
    return [conversation_key(place.provider, account_id or place.account_id, conversation, thread)
            for conversation in conversations for thread in threads]


def _call_account(ctx: Any, place: Any, targets: Mapping[str, list[str]]) -> str:
    account = next(iter(targets.get("account_id", [])), "")
    if account or place.account_id not in {"", "*"}:
        return account or place.account_id
    presence = _metadata(ctx).get("presence") or {}
    event = presence.get("event") or {}
    return str(event.get("account_id") or "") if presence.get("transport_skill") == place.connector else ""


def dispatch_refusal(ctx: Any, name: str, args: Mapping[str, Any], *, ext_tool: Any = None,
                     is_mcp: bool = False) -> str:
    """The refusal text for this call, ``""`` to allow."""
    from ouroboros.consciousness_authority import is_observe_origin

    try:
        effect, targets, resolved, siblings, attributable = _call_destinations(ctx, name, args, ext_tool, is_mcp)
    except _PlacesUnreadable as exc:
        effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx, args=args)
        if effect.effect == EFFECT_READ and not is_observe_origin(_metadata(ctx)):
            return ""
        return (_PREFIX + f"The place records need repair, so {name!r} cannot be attributed to a place: {exc}. "
                "Writes through this connector wait until Circles can read them again.")
    if not effect.effect:
        return ""
    if is_observe_origin(_metadata(ctx)) and not (effect.declared and effect.effect == EFFECT_READ):
        return (_PREFIX + f"Advises permits only declared reads from connectors and MCP; {name!r} may change "
                "something outside. Use a declared read tool, or change Consciousness to Acts or Evolves.")
    if effect.effect == EFFECT_READ:
        return ""
    for place in resolved:
        if place.state in (places.STATE_SILENT, places.STATE_PAUSED):
            return (_PREFIX + f"{place.label or place.place_id} is {place.state.title()}; {name!r} cannot write there. "
                    "Change its state in Circles or with configure_presence before writing there.")
    blocked = [place for place in siblings if place.state in (places.STATE_SILENT, places.STATE_PAUSED)]
    if not attributable and blocked:
        labels = ", ".join(place.label or place.place_id for place in blocked)
        return (_PREFIX + f"The destination of {name!r} cannot be attributed while {labels} is Silent or Paused. "
                "Use a tool with declared, attributable targets or change the state in Circles "
                "or with configure_presence.")
    return _author_refusal(ctx, name, args, effect, targets, resolved) if effect.effect == EFFECT_MESSAGE else ""


def _author_refusal(ctx: Any, name: str, args: Mapping[str, Any], effect: ToolEffect, targets: Mapping[str, list[str]],
                    resolved: list) -> str:
    """One author per conversation for words: another live turn's conversation names its author; this run's own
    sends there are noted (``_note_message_call``)."""
    keys = []
    for place in resolved:
        for key in _conversation_keys(place, effect, targets, account_id=_call_account(ctx, place, targets)):
            author = conversation_lease.live_author(key, drive_root=_drive_root(ctx))
            if author and author != getattr(ctx, "task_id", None):
                return (_PREFIX + f"Task {author} is authoring this conversation in {place.label or place.place_id}. "
                        f"Hand the intent to it with tell_task(task_id={author!r}, note=...).")
            keys.append(key)
    _note_message_call(ctx, name, args, keys, effect)
    return ""


def authored_notes(records: Any, place: Any) -> list[dict[str, Any]]:
    """A place's guidance as the Safety bundle and destination lines carry it, from one ``places.place_records`` read:
    its own notes, and its circle's words as one more entry with the circle's name and who placed the place there.
    ``standing_delegation`` is my human's standing consent and restriction: their own note, or their circle words
    through their own placement; through a placement I made their words still restrict, consenting to nothing, and
    words I wrote stay context."""
    entries = [{"place_id": place.place_id, "text": note.text, "author": note.author, **({"at": note.at} if note.at else {}),
                "standing_delegation": note.author in places.OWNER_VOICE} for note in place.notes]
    circle = records.circle_context(place.place_id)
    if circle and circle["text"]:
        entries.append({"place_id": place.place_id, "circle": circle["name"], "circle_id": circle["circle_id"],
                        "text": circle["text"], "author": circle["text_author"],
                        **({"at": circle["text_at"]} if circle["text_at"] else {}), "placed_by": circle["placed_by"],
                        "standing_delegation": circle["standing_delegation"]})
    return entries


def destination_line(label: str, entries: list[Mapping[str, Any]]) -> str:
    """One destination as the mind reads it beside an act or a refusal: its label, my human's notes for it and its
    circle's words, labelled with who wrote them and who placed the place there (``authored_notes``)."""
    import json

    def who(author: str) -> str:
        return "your human" if author in places.OWNER_VOICE else "you"

    def standing(entry: Mapping[str, Any]) -> str:
        # My human's words through a placement I made still restrict; their consent needs their own placement.
        if entry["standing_delegation"]:
            return ""
        if entry["author"] in places.OWNER_VOICE:
            return " (they still restrict here; consent needs your human's placement)"
        return " (context, not your human's choice for this place)"
    notes = [json.dumps(entry["text"], ensure_ascii=False) for entry in entries
             if "circle" not in entry and entry["standing_delegation"]]
    circles = [f"; its circle {json.dumps(entry['circle'], ensure_ascii=False)}'s words, by {who(entry['author'])}, "
               f"placed there by {who(entry['placed_by'])}" + standing(entry)
               + f": {json.dumps(entry['text'], ensure_ascii=False)}" for entry in entries if "circle" in entry]
    return f"[DESTINATION] {label}: your human's notes for this place: " + ("; ".join(notes) or "none") + "".join(circles)


def with_destination_facts(ctx: Any, name: str, args: Mapping[str, Any], result: Any) -> Any:
    """A send into a place other than this turn's focus carries that place's label, my human's notes and its circle's
    words beside its result, in every mode (my human's own work with Safety off included): the facts sit next to the
    act, after it, never as guidance the act had before it ran. ``result``
    is a ``ToolResult``; the note trails its text as a host annotation, the producer's text kept apart. A blocked
    call carries its own facts (a place refusal names the place, a Safety refusal its destination)."""
    try:
        if (not isinstance(args, Mapping) or getattr(result, "status", "") == "blocked"
                or effect_of(name, ctx=ctx, args=args).effect != EFFECT_MESSAGE):
            return result
        _effect, _targets, resolved, _siblings, _known = _call_destinations(ctx, name, args, None, False)
        others = [place.place_id for place in resolved if place.place_id != (_metadata(ctx).get("focus") or {}).get("place_id")]
        records = places.place_records(_drive_root(ctx)) if others else None
    except Exception:  # the call already ran: a missing note never turns its result into a failure
        import logging
        logging.getLogger(__name__).warning("destination facts for %s unavailable", name, exc_info=True)
        return result
    # Each destination's notes and circle words from one read of the store, the label it has now.
    note = "\n".join(destination_line(place.label or place.place_id, authored_notes(records, place))
                     for place in (records.places.get(key) for key in others) if place is not None)
    if not note:
        return result
    from dataclasses import replace
    return replace(result, text=f"{result.text}\n\n{note}", host_annotations=(*result.host_annotations, note),
                   producer_text=result.text if result.producer_text is None else result.producer_text)


def _note_message_call(ctx: Any, name: str, args: Mapping[str, Any], keys: list[str], effect: ToolEffect) -> None:
    """Keep on the context that this run dispatched a ``message`` call into these conversations (``_place_message_calls``):
    the words proposed (its declared ``text`` argument, default ``text``, when it has one) and when. A dispatched call is not a delivery:
    Safety may still refuse it and the handler may fail; the connector's receipt, when it comes, says what landed.
    ``this_turn_sends`` reads it beside the receipts, so a finish seconds after a send is judged with the send in view."""
    from ouroboros.utils import utc_now_iso

    if not keys:
        return
    calls = getattr(ctx, "_place_message_calls", None)
    if not isinstance(calls, list):
        calls = []
        try:
            ctx._place_message_calls = calls
        except AttributeError:  # a context that takes no attributes keeps no record
            return
    words = argument_value(args, effect.text or "text")
    calls.append({"tool": name, "at": utc_now_iso(), "conversation_keys": list(dict.fromkeys(keys)),
                  **({"text": words} if isinstance(words, str) else {})})


def this_turn_sends(ctx: Any, conversation_key: str) -> dict[str, Any]:
    """What this run already said in ``conversation_key``, as host facts: ``confirmed`` are the connector's receipts
    (``presence_send_facts``: delivered or accepted texts, uncertain parts, reactions, or an honest unknown);
    ``calls_without_receipt`` are this run's own dispatched ``message`` calls there whose words no receipt confirms
    yet (queued, refused by Safety or failed: the receipt, when it comes, says which)."""
    from ouroboros.presence_context import presence_send_facts, presence_send_record
    from ouroboros.tool_access import canonical_data_root

    presence = _metadata(ctx).get("presence")
    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        root = canonical_data_root(ctx)
        record = presence_send_record(root, task_id, presence)
        confirmed = presence_send_facts(root, task_id, presence)
    except Exception:
        record, confirmed = None, "unknown (receipts unreadable)"
    delivered = set(record["sent"]) if record else set()
    calls = [row for row in getattr(ctx, "_place_message_calls", None) or []
             if isinstance(row, Mapping) and conversation_key in (row.get("conversation_keys") or ())]
    pending = [{key: row[key] for key in ("tool", "at", "text") if key in row}
               for row in calls if row.get("text") is None or row["text"] not in delivered]
    return {"confirmed": confirmed, "calls_without_receipt": pending}


def place_evidence(ctx: Any, place: dict[str, Any], place_ids: set[str], tool: str,
                   args: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each named place's guidance for the Safety bundle, read now from one snapshot (``authored_notes``); the bundle's
    own place gets its current label and state. A ``configure_presence`` call naming a circle also carries that
    circle's words and its places (``place["circle"]``): a circle change is judged with what it changes. Only
    such a call reads the canonical store, so a check that names no place or circle needs no data root."""
    from ouroboros.tool_access import canonical_data_root

    circle_id = str(args.get("circle_id") or "") if tool == "configure_presence" else ""
    if not place_ids and not circle_id:
        return []
    records, notes = places.place_records(canonical_data_root(ctx)), []
    for record in records.places.values():
        if record.place_id not in place_ids:
            continue
        if record.place_id == place.get("place_id"):
            place.update(place_label=record.label, place_state=record.state)
        notes.extend(authored_notes(records, record))
    circle = records.circle(circle_id)
    if circle is not None:
        place["circle"] = {**{key: circle[key] for key in ("circle_id", "name", "text", "text_author")},
                           "places": [{"place_id": member.place_id, "place_label": member.label,
                                       "place_state": member.state,
                                       "placed_by": records.choices[member.place_id]["placed_by"]}
                                      for member in records.members(circle_id)]}
    return notes


def call_place_facts(ctx: Any, name: str, args: Mapping[str, Any], *, ext_tool: Any = None,
                     is_mcp: bool = False) -> dict[str, Any]:
    """Host facts about where a call lands, for the Supervisor's bundle.

    ``{"effect", "declared", "targets": {arg: [values]}, "attributable", "places": [{"place_id",
    "place_label", "place_state", "conversation_id" (its own, if any), "conversation_key", ...}]}``; ``{}`` when no
    place is involved.
    A place's notes are standing delegation: the bundle carries them once, beside the owner's rules.
    """
    try:
        effect, targets, resolved, siblings, attributable = _call_destinations(ctx, name, args, ext_tool, is_mcp)
    except _PlacesUnreadable as exc:
        return {"effect": "", "declared": False, "targets": {}, "attributable": False, "places": [],
                "places_unreadable": str(exc)}
    if name == "place_reply":
        effect = ToolEffect(EFFECT_MESSAGE, targets=("account_id", "conversation_id"),
                            conversation="conversation_id", thread="thread_id", declared=True, source="builtin")
        # The reply's route is its retained destination (``place_reply.reply_carrier``), thread included.
        destination = args.get("destination")
        targets = target_values(effect, destination if isinstance(destination, Mapping) else args)
        root = _drive_root(ctx)
        resolved = [place for place in places.load_places(root) if place.place_id == args.get("place_id")] if root else []
    if not resolved and not siblings:
        return {}
    rows = []
    for place in resolved:
        from ouroboros.place_arrivals import recent_identity
        account = _call_account(ctx, place, targets)
        focus = _metadata(ctx).get("focus") or {}
        same_focus = focus.get("place_id") == place.place_id
        identity = focus.get("identity") if same_focus else recent_identity(_drive_root(ctx), place.place_id, account)
        sending = (identity or {}).get("sends_as")
        keys = _conversation_keys(place, effect, targets, account_id=account)
        rows.append({"place_id": place.place_id, "place_label": place.label, "place_state": place.state,
                     **({"conversation_id": place.conversation_id} if place.conversation_id else {}),
                     **({"sends_as": dict(sending), "sends_as_owner": (sending.get("kind") == "user" and bool(focus.get("identity_owner")))
                         if same_focus else places.writes_as_owner(place, identity, account,
                                                                   _counted_accounts(ctx).get(place.place_id))}
                        if sending else {}),
                     "conversation_key": keys[0] if len(keys) == 1 else "", "conversation_keys": keys})
    facts = {"effect": effect.effect, "declared": effect.declared, "targets": targets,
             "attributable": attributable, "places": rows}
    own = str(((_metadata(ctx).get("presence") or {}).get("event") or {}).get("conversation_key") or "")
    # Speech into this turn's own conversation is judged with what the turn already said there.
    if effect.effect == EFFECT_MESSAGE and own and any(own in row["conversation_keys"] for row in rows):
        facts["this_turn_sends"] = this_turn_sends(ctx, own)
    return facts


def outbox_binding_facts(drive_root: Any, connector: str, destination: Mapping[str, str], *,
                         binding: Mapping[str, Any] | None = None, require_place: bool = False) -> dict[str, Any]:
    """Capture or recheck one connector's enqueue-time place binding, without writes.

    The snapshot records provenance, not a grant: delivery still checks current
    scopes. A removed bound place never turns into an unbound operation. Callers
    retain this snapshot with the queued effect, and never synthesize it for old
    rows or infer it from ``origin.kind=tool``.
    """
    target = {key: destination.get(key, "") for key in _DELIVERY_TARGET_FIELDS}
    if any(not isinstance(value, str) for value in target.values()) or not target["provider"] or not target["account_id"]:
        raise ValueError("A canonical provider and account_id are required for an outbox binding.")
    declared_targets = {key: [value] for key, value in target.items() if key != "provider" and value}
    if binding is not None:
        if (not isinstance(binding, Mapping) or type(binding.get("version")) is not int
                or binding["version"] != PLACE_BINDING_VERSION or binding.get("connector") != connector
                or any(binding.get(key) != value for key, value in target.items())
                or not isinstance(binding.get("place_id"), str)
                or type(binding.get("connected_revision")) is not int or binding["connected_revision"] < 0):
            raise ValueError("The retained outbox binding does not match this connector and destination.")
        if binding["place_id"]:
            try:
                original = places.get_place(drive_root, binding["place_id"])
            except places.PlaceError as exc:
                if exc.code != "place_not_found":
                    raise
                return {"allowed": False, "binding": dict(binding), "state": "disconnected",
                        "code": "place_disconnected", "label": "", "place_id": binding["place_id"]}
            if original.connector != connector or original.provider != target["provider"]:
                raise ValueError("The retained place belongs to a different connector or provider.")
            if original.connected_revision != binding["connected_revision"]:
                return {"allowed": False, "binding": dict(binding), "state": "disconnected",
                        "code": "place_disconnected", "label": original.label, "place_id": original.place_id}
            primary_scope = places.resolve_place(drive_root, connector=connector, binding_id=original.place_id, **target)
            declared_scope, _ = places.target_place([original], targets=declared_targets,
                                                    conversation="conversation_id", thread="thread_id")
            if primary_scope is None and declared_scope is None:
                return {"allowed": False, "binding": dict(binding), "state": "changed",
                        "code": "place_scope_changed", "label": original.label, "place_id": original.place_id}
            if original.state in (places.STATE_SILENT, places.STATE_PAUSED):
                return _outbox_decision(original, binding)
        elif require_place:
            raise ValueError("Place-bound delivery cannot use an unbound snapshot.")

    current, attributable = places.place_for_target(
        drive_root, connector=connector, provider=target["provider"], targets=declared_targets,
        conversation="conversation_id", thread="thread_id")
    if not target["conversation_id"] and (current is None or current.state == places.STATE_ACTIVE):
        # Opaque account holds supplement the outgoing matcher. A primary
        # Active place cannot erase a hold found through a declared destination.
        current = places.outbox_place(drive_root, connector=connector, **target) or current
    if current is None and binding and binding["place_id"]:
        return {"allowed": False, "binding": dict(binding), "state": "changed",
                "code": "place_scope_changed", "label": "", "place_id": binding["place_id"]}
    if current is None and (not attributable or require_place):
        return {"allowed": False, "binding": dict(binding) if binding else None, "state": "unknown",
                "code": "place_unattributable" if not attributable else "place_not_connected", "label": "", "place_id": ""}
    captured = dict(binding) if binding is not None else {
        "version": PLACE_BINDING_VERSION, "connector": connector, **target,
        "place_id": current.place_id if current else "", "connected_revision": current.connected_revision if current else 0}
    return _outbox_decision(current, captured)


def _outbox_decision(place: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    state = place.state if place else "unbound"
    return {"allowed": state in (places.STATE_ACTIVE, "unbound"), "binding": dict(binding),
            "state": state, "code": f"place_{state}", "label": place.label if place else "",
            "place_id": place.place_id if place else ""}
