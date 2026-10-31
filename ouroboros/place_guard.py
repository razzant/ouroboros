"""Dispatch-time refusals that follow from tool effects and place state.

The protected dispatch guard (``registry_guards._capability_resource_guard_result``)
asks ``dispatch_refusal`` once per call, the way it asks
``consciousness_authority.observe_argument_refusal``. Three rules live here:

* Advises (stored ``observe``): an extension or MCP call whose effect is not a
  declared read is refused, declared or not.
* Silent / Paused: a ``message`` or ``write`` aimed at such a place is refused
  from every focus (Main, tasks, schedules, wakes). A connector write whose
  target cannot be attributed is refused while any of that connector's places
  is Silent or Paused, and the reason says so.
* One author per conversation: a send into a conversation another turn is
  authoring is refused with that author's task id (hand the intent over with
  ``tell_task``).

Place lookup and lease ownership belong to their stores, not this policy.
"""

from __future__ import annotations

from typing import Any, Mapping

from ouroboros import places, conversation_lease
from ouroboros.tool_effects import EFFECT_MESSAGE, EFFECT_READ, ToolEffect, connector_of, effect_of, target_values

_PREFIX = "⚠️ RESOURCE_CONSTRAINT_BLOCKED: "


def _metadata(ctx: Any) -> Mapping[str, Any]:
    value = getattr(ctx, "task_metadata", {})
    return value if isinstance(value, Mapping) else {}


def _drive_root(ctx: Any) -> Any:
    return (_metadata(ctx).get("budget_drive_root") or getattr(ctx, "budget_drive_root", "")
            or getattr(ctx, "drive_root", None))


class _PlacesUnreadable(RuntimeError):
    """The place store needs repair; a connector's writes cannot be attributed."""


def _declares_place(connector: str) -> bool:
    """Whether the live publication of this skill declares a place (it is a connector)."""
    from ouroboros.extension_registry_state import live_contribution_projection

    return any(row.get("provides") == "place" and row.get("skill") == connector
               for row in live_contribution_projection())


def _call_destinations(ctx: Any, name: str, args: Mapping[str, Any], ext_tool: Any, is_mcp: bool):
    """Resolve each recipient separately so an Active To never hides a Silent Cc."""
    effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx)
    targets = target_values(effect, args)
    connector, root = connector_of(name, ext_tool), _drive_root(ctx)
    if not connector or root is None:
        return effect, targets, [], [], True
    try:
        siblings = places.places_of_connector(root, connector)
    except places.PlaceError as exc:
        # An unreadable place store never blocks skills that connect no place; a
        # connector's writes wait for the repair (``dispatch_refusal`` names it).
        if not _declares_place(connector):
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
        effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx)
        if effect.effect == EFFECT_READ and not is_observe_origin(_metadata(ctx)):
            return ""
        return (_PREFIX + f"The place records need repair, so {name!r} cannot be attributed to a place: {exc}. "
                "Writes through this connector wait until Consciousness › Places can read them again.")
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
                    "Change its state in Consciousness › Places or with configure_presence before writing there.")
    blocked = [place for place in siblings if place.state in (places.STATE_SILENT, places.STATE_PAUSED)]
    if not attributable and blocked:
        labels = ", ".join(place.label or place.place_id for place in blocked)
        return (_PREFIX + f"The destination of {name!r} cannot be attributed while {labels} is Silent or Paused. "
                "Use a tool with declared, attributable targets or change the state in Consciousness › Places "
                "or with configure_presence.")
    if effect.effect == EFFECT_MESSAGE:
        for place in resolved:
            for key in _conversation_keys(place, effect, targets, account_id=_call_account(ctx, place, targets)):
                author = conversation_lease.live_author(key, drive_root=_drive_root(ctx))
                if author and author != getattr(ctx, "task_id", None):
                    return (_PREFIX + f"Task {author} is authoring this conversation in {place.label or place.place_id}. "
                            f"Hand the intent to it with tell_task(task_id={author!r}, note=...).")
    return ""


def call_place_facts(ctx: Any, name: str, args: Mapping[str, Any], *, ext_tool: Any = None,
                     is_mcp: bool = False) -> dict[str, Any]:
    """Host facts about where a call lands, for the Supervisor's bundle.

    ``{"effect", "declared", "targets": {arg: [values]}, "attributable", "places": [{"place_id",
    "place_label", "place_state", "conversation_key", ...}]}``; ``{}`` when no place is involved.
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
                     **({"sends_as": dict(sending), "sends_as_owner": sending["kind"] == "user" and (bool(focus.get("identity_owner")) if same_focus
                         else places.is_owner_account(place, sending["id"], account))} if sending else {}),
                     "conversation_key": keys[0] if len(keys) == 1 else "", "conversation_keys": keys})
    return {"effect": effect.effect, "declared": effect.declared, "targets": targets,
            "attributable": attributable, "places": rows}
