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

A call reaches places through its declared target and its declared groups
(``tool_effects.touched_addresses``), resolved once here (``_call_destinations``)
for every reader: each group address alone, with its group's effect and match,
so a write or read beside words is never speech; a grouped row's own destination
is its conversation, else its account, while its other targets are its audience;
a call whose addresses cannot be read is refused whole. Place lookup and lease
ownership belong to their stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ouroboros import places, conversation_lease
from ouroboros.presence_bindings import conversation_key, direct_user_route, is_direct_id
from ouroboros.tool_effects import (EFFECT_MESSAGE, EFFECT_READ, ToolEffect, argument_value, bound_declaration,
                                    connector_of, direct_send_selectors, effect_of, target_values, thread_pairs, touched_addresses)

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


@dataclass
class _Reach:
    """One place a call reaches: the effects landing there, how (``target``: the declared target; ``account``: a
    grouped row's account-level operation, which names no conversation; ``touches``: a group) and the conversation
    keys of the call's own effect there (where words land, for a message)."""

    place: Any
    account: str = ""
    effects: set = field(default_factory=set)
    via: set = field(default_factory=set)
    keys: list = field(default_factory=list)


@dataclass
class _CallReach:
    """Where one call lands, resolved once (``_call_destinations``) for every reader."""

    effect: ToolEffect
    targets: dict
    touches: tuple = ()
    reaches: dict = field(default_factory=dict)  # (place_id, account) -> _Reach, in the order reached
    touched: dict = field(default_factory=dict)  # TouchAddress -> the place_id it reached
    siblings: list = field(default_factory=list)
    attributable: bool = True
    problem: str = ""
    direct_route: tuple[str, str, str] | None = None

    def reach(self, place: Any, effect: str, via: str, account: str, keys: list) -> None:
        found = self.reaches.setdefault((place.place_id, account), _Reach(place, account))
        found.effects.add(effect)
        found.via.add(via)
        found.keys.extend(key for key in keys if key not in found.keys)


def _call_destinations(ctx: Any, name: str, args: Mapping[str, Any], ext_tool: Any, is_mcp: bool) -> _CallReach:
    """Resolve each recipient and each group address separately, so an Active To never hides a Silent Cc and no
    address borrows another's account or thread. A call its connector declares as one that delivers words
    (``sends.when``) is a ``message`` here, routed by that declaration's selectors and groups."""
    effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx, args=args)
    targets = target_values(effect, args)
    touches, problem = touched_addresses(effect, args)
    call = _CallReach(effect, targets, touches, problem=effect.problem or problem)
    connector, root = connector_of(name, ext_tool), _drive_root(ctx)
    if call.problem or not connector or root is None:
        return call
    try:
        call.siblings = places.places_of_connector(root, connector)
    except places.PlaceError as exc:
        # An unreadable place store never blocks skills that connect no place; a
        # connector's writes wait for the repair (``dispatch_refusal`` names it).
        if not _declares_place(connector, ext_tool):
            return call
        raise _PlacesUnreadable(str(exc)) from exc
    if call.siblings and effect.effect != EFFECT_READ:
        # Several conversations beside several threads name no pairs: never a product of the two (a read only reads).
        call.problem = _target_pairs(effect, targets, None)[1]
        if call.problem:
            return call
    routing = {key: targets[key] for key in ("account_id", effect.conversation, effect.thread) if key in targets}
    selectors = direct_send_selectors(name, ext_tool)
    user_dialog = False
    if selectors:
        provider, pin, chat = selectors
        if provider == "telegram" and effect.thread and telegram_topic(argument_value(args, effect.thread)) is None:
            call.problem = "a Telegram topic needs one canonical positive decimal string ID"
            return call
        users = targets.get(chat, [])
        raw_chat = argument_value(args, chat)
        if (provider == "telegram" and len(users) == 1 and users[0].isascii() and users[0].isdecimal()
                and not is_direct_id(raw_chat, provider)):
            call.problem = "a Telegram user chat needs one canonical decimal string ID"
            return call
        user_dialog = len(users) == 1 and is_direct_id(users[0], provider)
        if user_dialog:
            call.siblings = [place for place in call.siblings if place.provider == provider]
            call.direct_route = direct_user_route(provider, args.get(pin), argument_value(args, chat))
            if call.direct_route:
                routing["account_id"] = [call.direct_route[1]]
            elif (_owner_direct_accounts(call.siblings, users[0])
                  or _direct_place_accounts(call.siblings, users[0], effect, targets)):
                call.problem = "this account-specific dialog needs an exact account pin"
                return call
    same = bool(effect.thread) and effect.thread == effect.conversation
    if effect.touches:
        destinations = _grouped_destinations(effect, targets, routing, args)
    else:  # a row without groups: each declared target value, else the whole call, which no value attributes
        destinations = [("target", key, {**routing, key: [value]}) for key in (effect.targets or (effect.conversation,))
                        for value in targets.get(key, [])]
        call.attributable = bool(destinations)
    for via, key, destination in destinations or [("target", "", targets)]:
        # One argument naming both conversation and thread pairs each id with itself; beside another key its
        # list is no thread, so no id is matched with another's thread.
        thread = "" if same and key != effect.conversation else effect.thread
        if user_dialog:
            place, known = places.target_place(call.siblings, targets=destination, conversation=effect.conversation, thread=thread)
        else:
            place, known = places.place_for_target(root, connector=connector, targets=destination,
                                                  conversation=effect.conversation, thread=thread)
        call.attributable = call.attributable and known
        if place is not None:
            addressed = destination if call.direct_route or via == "account" else targets
            account = next(iter(addressed.get("account_id", [])), "")
            if not user_dialog:
                account = _call_account(ctx, place, account)
            call.reach(place, effect.effect, via, account, _conversation_keys(
                place, effect, destination if effect.touches else targets, account_id=account))
    _reach_addresses(call, root, connector, ctx)
    return call


def _grouped_destinations(effect: ToolEffect, targets: Mapping[str, list[str]], routing: Mapping[str, list[str]],
                          args: Mapping[str, Any]) -> list[tuple[str, str, dict]]:
    """A grouped row's own destination, apart from its audience and its groups: each value of its declared
    conversation/thread pair; an empty conversation names none, so the account it acts in, matched as an
    account-level operation (``via`` "account"), else the whole call. Its other target values are its audience: never
    destination scopes here; their places apply only through the connector's own groups. A row declaring no
    conversation selector at all reads its target values as destinations, as before."""
    if effect.conversation:
        named = [("target", effect.conversation, {**routing, effect.conversation: [value],
                  **({effect.thread: [thread]} if effect.thread else {})})
                 for value, thread in _target_pairs(effect, targets, None)[0]]
    else:
        named = [("target", key, {**routing, key: [value]}) for key in effect.targets
                 if key not in {"account_id", effect.thread} for value in targets.get(key, [])]
    if named:
        return named
    account = targets.get("account_id") or [
        str(value).strip() for value in (argument_value(args, "account_id"),)
        if type(value) in (str, int) and str(value).strip()]
    return [("account", "account_id", {**routing, "account_id": account})] if account else [("target", "", dict(routing))]


def _reach_addresses(call: _CallReach, root: Any, connector: str, ctx: Any, provider: str = "", *,
                     as_routed: bool = False) -> None:
    """Each group address alone, by its group's ``match``: ``scope`` meets the destination matcher (inside an
    account's places an address none of them covers stays unknown, its account named or not; one outside every
    place's account is no place's), ``exact`` only a place naming it (none: a known absence). ``as_routed`` (an
    automatic reply's addresses) keeps each address's account, its route's, blank included, never inferring one."""
    for address in call.touches:
        target = {"conversation_id": [address.conversation], **({"account_id": [address.account]} if address.account
                                                                else {}),
                  **({"thread_id": [address.thread]} if address.thread else {})}
        place, known = places.place_for_target(root, connector=connector, targets=target, conversation="conversation_id",
                                             thread="thread_id", provider=provider, match=address.match)
        if address.effect != EFFECT_READ:
            call.attributable = call.attributable and known
        if place is not None:
            account = address.account if as_routed else _call_account(ctx, place, address.account)
            call.touched[address] = place.place_id
            call.reach(place, address.effect, "touches", account, [_key(place, account, address.conversation,
                                                                        address.thread)] if address.effect == call.effect.effect else [])


def _reply_addresses(call: _CallReach, ctx: Any, args: Mapping[str, Any], root: Any, place: Any) -> str:
    """The automatic final's further addresses: its selected anchor's literal ``message.reply_targets``, which the
    reply's subject carries as ``reply_anchor``, each met by its own match in the reply's verified route
    (``_reply_route``); why they cannot be read, else ``""``. A subject without them adds none (the legacy boundary)."""
    from ouroboros.presence_events import reply_targets

    anchor = args.get("reply_anchor")
    if anchor is None:
        return ""
    try:
        if not isinstance(anchor, Mapping):
            raise ValueError("reply_anchor must be an object")
        entries = reply_targets(anchor) or ()
    except ValueError as exc:
        return str(exc)
    provider, account = _reply_route(place, args.get("destination"))
    call.touches = _reply_touches(entries, account)
    _reach_addresses(call, root, place.connector, ctx, provider, as_routed=True)
    return ""


def _reply_route(place: Any, destination: Any) -> tuple[str, str]:
    """The provider and account an automatic reply and its further addresses live in: its verified route, the retained
    destination the reply goes to (``Place.destination`` where the place has one, else its own scope), never the
    source event's account; the connector is the reply place's. An unnamed or wildcard account is any account."""
    route = destination if isinstance(destination, Mapping) else {}
    account = str(route.get("account_id") or "")
    return str(route.get("transport") or "") or place.provider, "" if account == "*" else account


def direct_reply_route(presence: Any, place: places.Place, destination: Mapping) -> tuple[str, str, str] | None:
    """The connector's promised reply to this very private user, if it uses the retained route.

    The authenticated event's conversation (never its author) names the user. The connector promises
    to pin its own event's account, chat and thread on delivery. A redirected or legacy reply lacks this proof.
    """
    if (not isinstance(presence, Mapping) or presence.get("transport_skill") != place.connector
            or presence.get("place_id") != place.place_id):
        return None
    event = presence.get("event")
    if not isinstance(event, Mapping) or not presence.get("direct_reply_key"):
        return None
    account, user, thread = event.get("account_id"), event.get("conversation_id"), event.get("thread_id")
    key = conversation_key(place.provider, account, user, thread)
    if (presence["direct_reply_key"] != key or event.get("conversation_key") != key or not isinstance(thread, str)
            or (place.provider == "telegram" and telegram_topic(thread) != thread)
            or event.get("provider") != place.provider or event.get("destination") != destination
            or destination != {"transport": place.provider, "account_id": account, "conversation_id": user, "thread_id": thread}):
        return None
    return direct_user_route(place.provider, account, user)


def _direct_place_accounts(records: list[places.Place], user: str, effect: ToolEffect, targets: dict) -> set[str]:
    """Accounts naming this dialog. An account-wide fallback counts only if the ordinary matcher would select it."""
    fallback, _ = places.target_place(records, targets=targets, conversation=effect.conversation, thread=effect.thread)
    threads = targets.get(effect.thread, [])
    named, account_wide = set(), set()
    for place in records:
        scopes = [(place.account_id, place.conversation_id, place.thread_id)]
        if place.destination:
            scopes.append(tuple(place.destination.get(key, "") for key in ("account_id", "conversation_id", "thread_id")))
        for account, room, topic in scopes:
            if not topic or topic in threads:
                if room == user:
                    named.add(account)
                elif not room and place is fallback:
                    account_wide.add(account)
    return (named or account_wide) - {"", "*"}


def _owner_direct_accounts(records: Iterable[places.Place], user: str) -> set[str]:
    """Only current explicit owner associations in these connector/provider places count as evidence."""
    accounts = set()
    for place in records:
        for entry in place.owner_accounts:
            account = entry.account_id or place.account_id
            if entry.platform_actor_id == user and is_direct_id(account, place.provider):
                accounts.add(account)
    return accounts


def telegram_topic(value: Any) -> str | None:
    """Empty or an exact positive decimal topic; None means invalid."""
    if value is None or value == "":
        return ""
    return value if is_direct_id(value, "telegram") else None


def owner_dm_binding(records, connector, route, thread, *, tool="place_reply", expected=None):
    """The exact confirmation of this recipient, not the Place that governs delivery.

    Requiring the recorded entry distinguishes a later reconfirmation or reconnection.
    Another place of this bot may have supplied the confirmation.
    """
    if not route or route[0] != "telegram" or telegram_topic(thread) != thread:
        return None
    provider, account, user = route
    for place in records:
        if (place.connector, place.provider) != (connector, provider):
            continue
        for entry in place.owner_accounts:
            if not entry.confirmed_at or (entry.account_id or place.account_id, entry.platform_actor_id) != (account, user):
                continue
            binding = {"connector": connector, "tool": tool, "provider": provider, "account_id": account,
                       "conversation_id": user, "thread_id": thread, "place_id": place.place_id,
                       "connected_revision": place.connected_revision, "confirmed_at": entry.confirmed_at}
            if expected is None or binding == expected:
                return binding
    return None


def capture_owner_dm(root, snapshot, connector, route, thread, *, tool="place_reply"):
    """Capture the proof of audience and the selected outbound policy from one host snapshot."""
    identity = owner_dm_binding(snapshot.places.values(), connector, route, thread, tool=tool)
    if identity is None:
        return None
    target = {key: identity[key] for key in _DELIVERY_TARGET_FIELDS}
    decision = outbox_binding_facts(root, connector, target, require_place=True, snapshot=snapshot)
    if not decision["binding"]:
        return None
    condition = {**identity, "observed_revision": snapshot.revision, "delivery_place": decision["binding"]}
    return condition if valid_owner_dm(condition, connector, target) else None


def valid_owner_dm(condition, connector, target) -> bool:
    """Validate the opaque condition's shape and target; the current owner proof still has to be read."""
    if not isinstance(condition, dict) or condition.keys() != {
        *_DELIVERY_TARGET_FIELDS, "connector", "tool", "place_id", "connected_revision", "confirmed_at",
        "observed_revision", "delivery_place",
    }:
        return False
    if (not isinstance(target, Mapping) or target.keys() != set(_DELIVERY_TARGET_FIELDS)
            or any(not isinstance(value, str) for value in target.values())):
        return False
    provider, account, user, thread = (target[key] for key in _DELIVERY_TARGET_FIELDS)
    outbound = condition["delivery_place"]
    return (condition["connector"] == connector and isinstance(connector, str) and bool(connector)
            and provider == "telegram" and direct_user_route(provider, account, user) is not None
            and telegram_topic(thread) == thread
            and all(condition[key] == target[key] for key in _DELIVERY_TARGET_FIELDS)
            and all(isinstance(condition[key], str) and condition[key] for key in ("tool", "place_id", "confirmed_at"))
            and type(condition["connected_revision"]) is int and condition["connected_revision"] >= 0
            and isinstance(outbound, dict) and outbound.keys() == {
                *_DELIVERY_TARGET_FIELDS, "connector", "version", "place_id", "connected_revision"}
            and outbound["connector"] == connector
            and type(outbound["version"]) is int and outbound["version"] == PLACE_BINDING_VERSION
            and all(outbound[key] == target[key] for key in _DELIVERY_TARGET_FIELDS)
            and isinstance(outbound["place_id"], str) and bool(outbound["place_id"])
            and type(outbound["connected_revision"]) is int and outbound["connected_revision"] >= 0
            and type(condition["observed_revision"]) is int
            and condition["observed_revision"] >= max(condition["connected_revision"], outbound["connected_revision"]))


def check_owner_dm(root, connector, target, condition, *, snapshot=None):
    """None means the original owner confirmation is gone; otherwise return the outbound Place decision."""
    if not valid_owner_dm(condition, connector, target):
        raise ValueError("The owner confirmation does not match this delivery address.")
    floor = condition["observed_revision"]
    if snapshot is None:
        snapshot = places.place_records(root, minimum_revision=floor)
    if snapshot.revision < floor:
        raise places.PlaceError("places_unreadable", "revision", "The Places snapshot predates the recorded check.")
    route = direct_user_route("telegram", target["account_id"], target["conversation_id"])
    identity = {key: value for key, value in condition.items() if key not in {"observed_revision", "delivery_place"}}
    if owner_dm_binding(snapshot.places.values(), connector, route, target["thread_id"],
                        tool=condition["tool"], expected=identity) is None:
        return None
    outbound = condition["delivery_place"]
    decision = outbox_binding_facts(root, connector, target, binding=outbound, require_place=True, snapshot=snapshot)
    if decision["place_id"] != outbound["place_id"]:
        return {**decision, "allowed": False, "state": "changed", "code": "place_scope_changed"}
    if not decision["allowed"]:
        return decision
    # An outbound declaration can strengthen the check, but cannot open an address the ordinary
    # named lookup holds. This check uses the same snapshot and never substitutes its Place for the binding.
    try:
        prior = places.resolve_place(root, connector=connector, places=snapshot.places.values(), **target)
    except places.PlaceError as exc:
        if exc.code != "place_ambiguous":
            raise
        return {**decision, "allowed": False, "state": "changed", "code": exc.code}
    if prior is None or prior.state != places.STATE_ACTIVE:
        code = "place_not_connected" if prior is None else f"place_{prior.state}"
        return {**decision, "allowed": False, "state": "changed", "code": code}
    return decision


def bind_direct_send(ctx: Any, name: str, ext_tool: Mapping, args: dict) -> bool:
    """Bind the host's private copy of this call before place policy, Safety and dispatch.
    A pin constrains a delivery. Only the connector can establish which account it actually has.
    Return whether these are a direct-recipient tool's effective arguments, for its later readers.
    """
    selectors = direct_send_selectors(name, ext_tool)
    if selectors is None:
        return False
    provider, pin, chat = selectors
    root = _drive_root(ctx)
    user = argument_value(args, chat)
    conditional = bound_declaration(ext_tool)["tool"]["direct_recipient"].get("confirmation") == "owner_confirmation"
    if conditional:
        args.pop("owner_confirmation", None)  # only this host can attach the delivery condition
    if root and (conditional or pin not in args) and is_direct_id(user, provider):
        try:
            skill = ext_tool["skill"]
            snapshot = places.place_records(root)
            records = [p for p in snapshot.places.values() if (p.connector, p.provider) == (skill, provider)]
            effect = effect_of(name, ext_tool=ext_tool, ctx=ctx, args=args)
            if pin not in args:
                presence = _metadata(ctx).get("presence") or {}
                event = presence.get("event") or {}
                focus = next((p for p in records if p.place_id == presence.get("place_id")), None)
                source = (direct_reply_route(presence, focus, event.get("destination") or {})
                          if focus and provider == "telegram" else None)
                if source and source[2] == user:
                    args[pin] = source[1]
                else:
                    accounts = (_owner_direct_accounts(records, user)
                                or _direct_place_accounts(records, user, effect, target_values(effect, args)))
                    if len(accounts) == 1:
                        account, = accounts
                        if is_direct_id(account, provider):
                            args[pin] = account
            if conditional:
                route = direct_user_route(provider, args.get(pin), user)
                topic = telegram_topic(argument_value(args, effect.thread)) if effect.thread else ""
                if topic is not None:
                    binding = capture_owner_dm(root, snapshot, skill, route, topic, tool=name)
                    if binding:
                        args["owner_confirmation"] = binding
        except places.PlaceError:
            pass  # dispatch_refusal reports the unreadable store; it cannot supply a pin.
    return True


def _reply_touches(entries: Any, account: str) -> tuple:
    from ouroboros.tool_effects import TouchAddress

    return tuple(TouchAddress(entry["effect"], entry["conversation"], account, entry.get("thread", ""), "reply_targets",
                              entry["match"]) for entry in entries)


def reply_address_hold(root: Any, place: Any, destination: Any, addresses: Mapping[str, Any] | None) -> str:
    """Why the automatic reply's further addresses (``place_reply.reply_addresses``) hold it now, ``""`` when none
    does: one meeting a Silent or Paused place by its match, or one the scope matcher cannot attribute while a place
    of the connector is Silent or Paused, the rule its dispatch would meet. Resolved now, never from stored states."""
    if not addresses or not addresses.get("reply_targets"):
        return ""
    provider, account = _reply_route(place, destination)
    call = _CallReach(ToolEffect(effect="", source="builtin"), {}, _reply_touches(addresses["reply_targets"], account))
    _reach_addresses(call, root, place.connector, None, provider, as_routed=True)
    for reach in call.reaches.values():
        if reach.place.state in (places.STATE_SILENT, places.STATE_PAUSED):
            return f"Reply held: {reach.place.label or reach.place.place_id} is {reach.place.state.title()}"
    blocked = [] if call.attributable else [other for other in places.places_of_connector(root, place.connector)
                                            if other.state in (places.STATE_SILENT, places.STATE_PAUSED)]
    if blocked:
        return ("Reply held: an address it reaches cannot be attributed while "
                + ", ".join(other.label or other.place_id for other in blocked) + " is Silent or Paused")
    return ""


def _key(place: Any, account: str, conversation: str, thread: str) -> str:
    return conversation_key(place.provider, account or place.account_id, conversation, thread)


def _target_pairs(effect: Any, targets: Mapping[str, list[str]], place: Any) -> tuple[list, str]:
    """The declared target's conversations, each with its own thread (``tool_effects.thread_pairs``); the place's
    own conversation when the call names none (``place`` None: only whether they pair)."""
    named = targets.get(effect.conversation, []) if effect.conversation else []
    threads = targets.get(effect.thread, []) if effect.thread else []
    same = bool(named) and bool(effect.thread) and effect.thread == effect.conversation
    own = [place.conversation_id] if place is not None and place.conversation_id else []
    return thread_pairs(named or own, threads, same_argument=same)


def _conversation_keys(place: Any, effect: Any, targets: Mapping[str, list[str]], *, account_id: str = "") -> list[str]:
    return [_key(place, account_id, conversation, thread) for conversation, thread in _target_pairs(effect, targets, place)[0]]


def _call_account(ctx: Any, place: Any, account: str = "") -> str:
    return _account_for(_metadata(ctx), place, account)


def _account_for(metadata: Mapping[str, Any], place: Any, account: str = "") -> str:
    if account or place.account_id not in {"", "*"}:
        return account or place.account_id
    presence = metadata.get("presence") or {}
    event = presence.get("event") or {}
    return str(event.get("account_id") or "") if presence.get("transport_skill") == place.connector else ""


def dispatch_refusal(ctx: Any, name: str, args: Mapping[str, Any], *, ext_tool: Any = None,
                     is_mcp: bool = False) -> str:
    """The refusal text for this call, ``""`` to allow."""
    from ouroboros.consciousness_authority import is_observe_origin

    try:
        call = _call_destinations(ctx, name, args, ext_tool, is_mcp)
    except _PlacesUnreadable as exc:
        effect = effect_of(name, ext_tool=ext_tool, is_mcp=is_mcp, ctx=ctx, args=args)
        if effect.effect == EFFECT_READ and not is_observe_origin(_metadata(ctx)):
            return ""
        return (_PREFIX + f"The place records need repair, so {name!r} cannot be attributed to a place: {exc}. "
                "Writes through this connector wait until Circles can read them again.")
    effect = call.effect
    if call.problem:
        repair = ("Its connector's declaration needs a fix (CREATING_SKILLS, \"Declaring a place, tool effects and "
                  "an icon\"), published again, before it runs." if effect.problem else
                  "Call it again with each address argument as declared (string ids; one account and one thread "
                  "beside each list of conversations), or split it.")
        return _PREFIX + f"Where {name!r} lands cannot be read: {call.problem}. Nothing ran. {repair}"
    if not effect.effect:
        return ""
    if is_observe_origin(_metadata(ctx)) and not (effect.declared and effect.effect == EFFECT_READ):
        return (_PREFIX + f"Advises permits only declared reads from connectors and MCP; {name!r} may change "
                "something outside. Use a declared read tool, or change Consciousness to Acts or Evolves.")
    if effect.effect == EFFECT_READ:
        return ""
    for reach in call.reaches.values():
        place = reach.place
        if reach.effects - {EFFECT_READ} and place.state in (places.STATE_SILENT, places.STATE_PAUSED):
            return (_PREFIX + f"{place.label or place.place_id} is {place.state.title()}; {name!r} cannot write there. "
                    "Change its state in Circles or with configure_presence before writing there.")
    blocked = [place for place in call.siblings if place.state in (places.STATE_SILENT, places.STATE_PAUSED)
               and (not call.direct_route or (place.provider == call.direct_route[0]
                                              and places.account_in_place_scope(place, call.direct_route[1])))]
    if not call.attributable and blocked:
        labels = ", ".join(place.label or place.place_id for place in blocked)
        return (_PREFIX + f"The destination of {name!r} cannot be attributed while {labels} is Silent or Paused. "
                "Use a tool with declared, attributable targets or change the state in Circles "
                "or with configure_presence.")
    return _author_refusal(ctx, name, args, call) if effect.effect == EFFECT_MESSAGE else ""


def _author_refusal(ctx: Any, name: str, args: Mapping[str, Any], call: _CallReach) -> str:
    """One author per conversation for words: another live turn's conversation names its author; this run's own
    sends there are noted (``_note_message_call``). Only where the words land holds an author: a group's write or
    read beside them takes none."""
    keys = []
    for reach in call.reaches.values():
        for key in reach.keys:
            author = conversation_lease.live_author(key, drive_root=_drive_root(ctx))
            if author and author != getattr(ctx, "task_id", None):
                place = reach.place
                return (_PREFIX + f"Task {author} is authoring this conversation in {place.label or place.place_id}. "
                        f"Hand the intent to it with tell_task(task_id={author!r}, note=...).")
            keys.append(key)
    _note_message_call(ctx, name, args, keys, call.effect)
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


def with_destination_facts(ctx: Any, name: str, args: Mapping[str, Any], result: Any, *, ext_tool: Any = None) -> Any:
    """A send into a place other than this turn's focus carries that place's label, my human's notes and its circle's
    words beside its result, in every mode (my human's own work with Safety off included): the facts sit next to the
    act, after it, never as guidance the act had before it ran. ``result``
    is a ``ToolResult``; the note trails its text as a host annotation, the producer's text kept apart. A blocked
    refusal before dispatch names its own destination. A host-refused further step can finish a dispatched call;
    its destination line describes the target, not whether an earlier part was delivered. ``ext_tool`` is
    the descriptor the call was dispatched with, so a publication that replaced it meanwhile never redescribes the
    act. Every place the send changes counts, a group's write beside the words included; a group's read does not."""
    try:
        status = getattr(result, "status", "")
        meta = getattr(result, "meta", None)
        started_refusal = (status == "blocked" and isinstance(meta, Mapping)
                           and meta.get("refused_step") is True and meta.get("physical_dispatch") is True)
        if (not isinstance(args, Mapping) or (status == "blocked" and not started_refusal)
                or effect_of(name, ext_tool=ext_tool, ctx=ctx, args=args).effect != EFFECT_MESSAGE):
            return result
        call = _call_destinations(ctx, name, args, ext_tool, False)
        focus = (_metadata(ctx).get("focus") or {}).get("place_id")
        others = dict.fromkeys(reach.place.place_id for reach in call.reaches.values()
                               if reach.effects - {EFFECT_READ} and reach.place.place_id != focus)
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
                   producer_text=(result.producer_text if result.host_annotations or result.producer_text is not None else result.text))


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


def _current_owner_dm(root, connector, route, thread, tool, binding):
    if not root or not route or not isinstance(binding, dict) or binding.get("tool") != tool:
        return None
    provider, account, user = route
    target = {"provider": provider, "account_id": account, "conversation_id": user, "thread_id": thread}
    try:
        decision = check_owner_dm(root, connector, target, binding)
    except ValueError:  # PlaceError is a ValueError: an unread store supplies no positive audience fact.
        return None
    return binding if decision and decision["allowed"] else None


def call_place_facts(ctx: Any, name: str, args: Mapping[str, Any], *, ext_tool: Any = None,
                     is_mcp: bool = False) -> dict[str, Any]:
    """Host facts about where a call lands, for the Supervisor's bundle.

    ``{"effect", "declared", "targets": {arg: [values]}, "attributable", "places": [{"place_id",
    "place_label", "place_state", "conversation_id" and "thread_id" (its own, if any), "account_id" (the account the
    call acts with there, if known), "conversation_key", ...}]}``; ``{}`` when no place is involved. A call with group addresses also carries ``touches`` (each exact address with its effect,
    argument, ``match`` when exact, and the place it reached), and a grouped call's rows their ``effects`` and ``via``;
    its ``conversation_keys`` name only where the call's own effect lands. An automatic final's retained destination
    owns its key and sending account; a blank account has no known sending identity. The place's scope and notes
    remain its own. Calls without a destination keep the legacy scope inference. The automatic final's further addresses
    (its anchor's ``message.reply_targets``, carried in the reply's subject) are such ``write`` groups. A call whose
    addresses cannot be read carries ``problem`` and no place.
    ``confirmed_owner_dm``, when present, names the sole external recipient if the connector sends.
    A place's notes are standing delegation: the bundle carries them once, beside the owner's rules.
    """
    try:
        call = _call_destinations(ctx, name, args, ext_tool, is_mcp)
    except _PlacesUnreadable as exc:
        return {"effect": "", "declared": False, "targets": {}, "attributable": False, "places": [],
                "places_unreadable": str(exc)}
    effect, targets = call.effect, call.targets
    if call.problem:
        return {"effect": effect.effect, "declared": False, "targets": {}, "attributable": False, "places": [],
                "problem": call.problem}
    destination = args.get("destination") if name == "place_reply" else None
    routed_reply = isinstance(destination, Mapping)
    binding = args.get("owner_confirmation")
    owner_dm = None
    if name == "place_reply":
        effect = ToolEffect(EFFECT_MESSAGE, targets=("account_id", "conversation_id"),
                            conversation="conversation_id", thread="thread_id", declared=True, source="builtin")
        # The reply's route is its retained destination (``place_reply.reply_carrier``), thread included.
        targets = target_values(effect, destination if routed_reply else args)
        root = _drive_root(ctx)
        call.reaches = {}
        records = places.load_places(root) if root else []
        for place in records:
            if place.place_id == args.get("place_id"):
                if routed_reply:
                    provider, account = _reply_route(place, destination)
                    keys = [conversation_key(provider, account, conversation, thread)
                            for conversation, thread in _target_pairs(effect, targets, None)[0]]
                else:
                    account = _call_account(ctx, place, next(iter(targets.get("account_id", [])), ""))
                    keys = _conversation_keys(place, effect, targets, account_id=account)
                call.reach(place, EFFECT_MESSAGE, "target", account, keys)
                if problem := _reply_addresses(call, ctx, args, root, place):
                    return {"effect": EFFECT_MESSAGE, "declared": False, "targets": {}, "attributable": False,
                            "places": [], "problem": problem}
                anchor = args.get("reply_anchor")
                presence = _metadata(ctx).get("presence") or {}
                if (isinstance(binding, dict) and presence.get("direct_reply_version") == 2 and routed_reply
                        and isinstance(anchor, Mapping) and anchor.get("reply_targets") == []):
                    route = direct_reply_route(presence, place, destination)
                    owner_dm = _current_owner_dm(root, place.connector, route, destination.get("thread_id"), name, binding)
    elif (isinstance(binding, dict) and call.direct_route and effect.declared
          and effect.effect == EFFECT_MESSAGE and not call.touches
          and bound_declaration(ext_tool)["tool"]["direct_recipient"].get("confirmation") == "owner_confirmation"):
        topic = telegram_topic(argument_value(args, effect.thread)) if effect.thread else ""
        owner_dm = _current_owner_dm(_drive_root(ctx), connector_of(name, ext_tool), call.direct_route, topic, name, binding)
    if not call.reaches and not call.siblings and not call.touches:
        return {}
    rows = []
    conditional_sender = (direct_send_selectors(name, ext_tool) is not None
                          or (routed_reply and (_metadata(ctx).get("presence") or {}).get("direct_reply_key")))
    for reach in call.reaches.values():
        from ouroboros.place_arrivals import recent_identity
        place, account, keys = reach.place, reach.account, reach.keys
        focus = _metadata(ctx).get("focus") or {}
        # A proposed pin specifies the only possible account. A prior identity is no observation of this client.
        known_sender = not conditional_sender and (
            not routed_reply or (bool(account) and _reply_route(place, destination)[0] == place.provider))
        same_focus = known_sender and focus.get("place_id") == place.place_id and account == _call_account(ctx, place)
        identity = (focus.get("identity") if same_focus else recent_identity(_drive_root(ctx), place.place_id, account)
                    if known_sender else None)
        sending = (identity or {}).get("sends_as")
        rows.append({"place_id": place.place_id, "place_label": place.label, "place_state": place.state,
                     **({"conversation_id": place.conversation_id} if place.conversation_id else {}),
                     **({"thread_id": place.thread_id} if place.thread_id else {}),
                     **({"account_id": account} if account and account != "*" else {}),
                     **({"sends_as": dict(sending), "sends_as_owner": (sending.get("kind") == "user" and bool(focus.get("identity_owner")))
                         if same_focus else places.writes_as_owner(place, identity, account,
                                                                   _counted_accounts(ctx).get(place.place_id))}
                        if sending else {}),
                     "conversation_key": keys[0] if len(keys) == 1 else "", "conversation_keys": list(keys),
                     **({"effects": sorted(reach.effects), "via": sorted(reach.via)}
                        if call.touches or call.effect.touches else {})})
    facts = {"effect": effect.effect, "declared": effect.declared, "targets": targets,
             "attributable": call.attributable, "places": rows}
    if owner_dm:
        facts["confirmed_owner_dm"] = {"connector": owner_dm["connector"], "provider": owner_dm["provider"],
                                       "account_id": owner_dm["account_id"], "user_id": owner_dm["conversation_id"]}
    if call.touches:
        facts["touches"] = [{"effect": address.effect, "argument": address.argument, "conversation": address.conversation,
                             **({"thread": address.thread} if address.thread else {}),
                             **({"account": address.account} if address.account else {}),
                             **({"match": address.match} if address.match != places.MATCH_SCOPE else {}),
                             **({"place_id": call.touched[address]} if address in call.touched else {})}
                            for address in call.touches]
    own = str(((_metadata(ctx).get("presence") or {}).get("event") or {}).get("conversation_key") or "")
    # Speech into this turn's own conversation is judged with what the turn already said there.
    if effect.effect == EFFECT_MESSAGE and own and any(own in row["conversation_keys"] for row in rows):
        facts["this_turn_sends"] = this_turn_sends(ctx, own)
    return facts


def outbox_binding_facts(drive_root: Any, connector: str, destination: Mapping[str, str], *,
                         binding: Mapping[str, Any] | None = None, require_place: bool = False,
                         match: str = places.MATCH_SCOPE,
                         snapshot: places.PlaceRecords | None = None) -> dict[str, Any]:
    """Capture or recheck one connector's enqueue-time place binding.

    For a named destination a caller's Places snapshot supplies both the original
    and the currently selected Place without rereading policy between them.
    The binding records provenance, not a grant: delivery still checks current
    scopes. A removed bound place never turns into an unbound operation. Callers
    retain this snapshot with the queued effect, and never synthesize it for old
    rows or infer it from ``origin.kind=tool``. ``match`` is how the destination meets
    places (``places.target_place``): ``scope``, or ``exact`` for an extra address
    constraint, which needs the conversation, has no opaque or unknown fallback and
    finds no place as a confirmed unbound one. The v1 snapshot never records it: the
    caller keeps it beside the snapshot and checks with the same mode.
    """
    if match not in places.MATCHES:
        raise ValueError(f"match is {' or '.join(places.MATCHES)}.")
    target = {key: destination.get(key, "") for key in _DELIVERY_TARGET_FIELDS}
    if any(not isinstance(value, str) for value in target.values()) or not target["provider"] or not target["account_id"]:
        raise ValueError("A canonical provider and account_id are required for an outbox binding.")
    exact = match != places.MATCH_SCOPE  # ``exact`` or ``exact_casefold``
    if exact and not target["conversation_id"]:
        raise ValueError("An exact place binding needs its conversation_id.")
    if snapshot is not None and not target["conversation_id"]:
        raise ValueError("A snapshot-based place check needs its conversation_id.")
    declared_targets = {key: [value] for key, value in target.items() if key != "provider" and value}
    if binding is not None:
        if (not isinstance(binding, Mapping) or type(binding.get("version")) is not int
                or binding["version"] != PLACE_BINDING_VERSION or binding.get("connector") != connector
                or any(binding.get(key) != value for key, value in target.items())
                or not isinstance(binding.get("place_id"), str)
                or type(binding.get("connected_revision")) is not int or binding["connected_revision"] < 0):
            raise ValueError("The retained outbox binding does not match this connector and destination.")
        if binding["place_id"]:
            if snapshot is not None:
                original = snapshot.places.get(binding["place_id"])
            else:
                try:
                    original = places.get_place(drive_root, binding["place_id"])
                except places.PlaceError as exc:
                    if exc.code != "place_not_found":
                        raise
                    original = None
            if original is None:
                return {"allowed": False, "binding": dict(binding), "state": "disconnected",
                        "code": "place_disconnected", "label": "", "place_id": binding["place_id"]}
            if original.connector != connector or original.provider != target["provider"]:
                raise ValueError("The retained place belongs to a different connector or provider.")
            if original.connected_revision != binding["connected_revision"]:
                return {"allowed": False, "binding": dict(binding), "state": "disconnected",
                        "code": "place_disconnected", "label": original.label, "place_id": original.place_id}
            if exact:  # its own primary or destination scope must still name this conversation: no broad pass
                primary_scope = None
                declared_scope, _ = places.target_place([original], targets=declared_targets, conversation="conversation_id",
                                                        thread="thread_id", match=match)
            else:
                primary_scope = places.resolve_place(drive_root, connector=connector, binding_id=original.place_id,
                                                     places=(original,) if snapshot is not None else None, **target)
                declared_scope, _ = places.target_place([original], targets=declared_targets,
                                                        conversation="conversation_id", thread="thread_id")
            if primary_scope is None and declared_scope is None:
                return {"allowed": False, "binding": dict(binding), "state": "changed",
                        "code": "place_scope_changed", "label": original.label, "place_id": original.place_id}
            if original.state in (places.STATE_SILENT, places.STATE_PAUSED):
                return _outbox_decision(original, binding)
        elif require_place:
            raise ValueError("Place-bound delivery cannot use an unbound snapshot.")

    if snapshot is None:
        current, attributable = places.place_for_target(
            drive_root, connector=connector, provider=target["provider"], targets=declared_targets,
            conversation="conversation_id", thread="thread_id", match=match)
    else:
        siblings = [place for place in snapshot.places.values()
                    if (place.connector, place.provider) == (connector, target["provider"])]
        current, attributable = places.target_place(
            siblings, targets=declared_targets, conversation="conversation_id", thread="thread_id", match=match)
    if not exact and not target["conversation_id"] and (current is None or current.state == places.STATE_ACTIVE):
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
