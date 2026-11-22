"""Situation names and ordinary host facts from place and room evidence.

Authorship comes from connector facts, never a provider name. Only the small
source openings used by event forms survive capture; readers retain the words.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ouroboros import chat_chain, dialogue_store
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.consciousness_authority import FOCUS_ROUTE_FIELDS, FOCUS_TRANSPORT_KEY, is_own_clock_wake
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID
from ouroboros.dialogue_provenance import RoomLabelResolver, _derives, recorded_person, recorded_speaker
from ouroboros.store import StoreError

WAKE = "wake"  # the name every wake my own clock starts has, beside Main's room; it stays with the wake, never a helper


def starts_conversation(author):
    """A recorded living person or an outward reply; an own-account arrival is neither."""
    return (recorded_person(author) and not author.get("is_bot") or
            author.get("kind") == "ouroboros" and author.get("via") != "own_account")


def capture(spoken):
    """Select samples and arrival bounds from entries already in hand, never enumerate/copy their bodies.

    Bounds and counts address an aggregated selection, not complete membership:
    a range reader can include non-arrivals. Legacy captures without bounds stay unknown.
    Arrivals without a recorded author are counted here too (owner choice A).
    """
    first = last = first_text = last_known = first_unknown = last_unknown = None
    count = unknown_count = unknown_chars = 0
    for entry, cls in spoken:
        _, row, pos = entry
        if cls["lane"] != 1 or row.get("direction") != "in":
            continue
        count += 1
        if first is None or pos < first[2]:
            first = entry
        if last is None or pos > last[2]:
            last = entry
        if row.get("text_chars") and (first_text is None or pos < first_text[0][2]):
            first_text = (entry, cls["author"])
        if recorded_speaker(row):
            if last_known is None or pos > last_known[0][2]:
                last_known = (entry, cls["author"])
        else:
            unknown_count += 1
            unknown_chars += int(row.get("text_chars") or 0)
            if first_unknown is None or pos < first_unknown[0][2]:
                first_unknown = (entry, cls["author"])
            if last_unknown is None or pos > last_unknown[2]:
                last_unknown = entry
    return {"count": count, "unknown_count": unknown_count,
            "bounds": [{"address": chat_chain.format_address(e[0]), "pos": e[2],
                        "ts": str(e[1].get("ts") or "")} for e in (first, last) if e is not None],
            "unknown_chars": unknown_chars, "unknown_last": str(last_unknown[1].get("ts") or "") if last_unknown else "",
            "_samples": {"first": first_text, "last": last_known, "unknown_first": first_unknown}}


def text_entries(event):
    return [pair[0] for pair in event["_samples"].values() if pair is not None]


def finish_capture(event, texts):
    """Replace transient metadata references with the scalar facts printed by the forms."""
    for name, pair in event.pop("_samples").items():
        if pair is None:
            event[name] = None
            continue
        entry, author = pair
        row = texts.get(entry[0]["row_sha256"], {})
        text = str(row.get("text") or "").strip()
        label = ("author not recorded" if author.get("unattributed") else author.get("label", "author not recorded"))
        if author.get("via") == "own_account":
            label += " (account not confirmed as my human's)"
        event[name] = {"address": chat_chain.format_address(entry[0]), "pos": entry[2],
                       "ts": str(entry[1].get("ts") or ""), "author": label,
                       "opening": " ".join(text.splitlines()[:1])[:160]}


def _sample(sample, *, opening=False):
    if sample is None:
        return "author not recorded"
    return f"[{sample['address']}; {sample['author']}]" + (f" «{sample['opening']}»" if opening else "")


def _unknown(event):
    if not event["unknown_count"]:
        return ""
    # Keep this one line even when the source's first 160 characters contain newlines.
    opening = " ".join((event["unknown_first"] or {}).get("opening", "").split())
    return (f"; author not recorded: {event['unknown_count']} arrivals, {event['unknown_chars']} chars, "
            f"last {event['unknown_last']}; earliest opening «{opening}»")


def room_line(room):
    from ouroboros.memory_view import _minute
    event = room["events"]
    if not event["count"]:
        last = f"; last {_minute(latest)}" if (latest := room["facts"].get("latest")) else ""
        return f"- {room['label']}; no arrivals{last}; memory_read(room_id='{room['room_id']}', rows=true)"
    return (f"- {room['label']}; Event: {event['count']} arrivals; first {_sample(event['first'], opening=True)}; "
            f"last {_sample(event['last'])}{_unknown(event)}; memory_read(room_id='{room['room_id']}', rows=true)")


def totals(rooms):
    events = [r["events"] for r in rooms if r.get("events")]
    return len(events), sum(e["count"] for e in events)


def count_line(group):
    events = [r["events"] for r in group["rooms"] if r.get("events")]
    if not events:
        return ""
    known = [e["last"] for e in events if e["last"]]
    unknown = [e["unknown_first"] for e in events if e["unknown_first"]]
    combined = {"unknown_count": sum(e["unknown_count"] for e in events),
                "unknown_chars": sum(e["unknown_chars"] for e in events),
                "unknown_last": max(e["unknown_last"] for e in events),
                "unknown_first": min(unknown, key=lambda s: s["pos"], default=None)}
    last = max(known, key=lambda s: s["pos"], default=None)
    return (f"Events: {len(events)} rooms, {sum(e['count'] for e in events)} arrivals; "
            f"last {_sample(last, opening=True)}{_unknown(combined)}; "
            f"memory_read(room_id='*', start={group['catalogue_start']})")


def catalogue_lines(group):
    """The exact unwindowed room catalogue, shared with its character-offset reader."""
    from ouroboros import memory_view as mv
    from ouroboros.memory_view_conversation_forms import identity_line
    def presence(room):
        return ("a person spoke here; no word of my human or reply of mine" if room.get("overheard") else
                "a person or I spoke here" if room["engaged"] else "no recorded conversation here")
    return [identity_line(group), *(
        f"  - {room['label']}; {presence(room)}; "
        f"{mv._facts_text(room['facts'], 'that room')}; memory_read(room_id='{room['room_id']}', rows=true)"
        for room in group["rooms"])]


def locate_catalogue(groups):
    """Offsets include the focus's conversation and room; filter that room only afterwards."""
    offset = 0
    for group in groups:
        group["catalogue_start"] = offset
        offset += sum(len(line) + 1 for line in catalogue_lines(group))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _place_facts(task: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    metadata = _mapping(task.get("metadata"))
    focus = _mapping(metadata.get("focus") or task.get("focus"))
    presence = _mapping(metadata.get("presence"))
    place_id = focus.get("place_id") or presence.get("place_id")
    route = (_mapping(presence.get("event")) or _mapping(metadata.get(FOCUS_TRANSPORT_KEY))
             or _mapping(focus.get("transport")))
    if not route and place_id:
        # Older carried focuses kept just the concurrency key. It yields exact
        # components only where its separators are unambiguous.
        parts = str(focus.get("conversation_key") or "").split(":")
        if len(parts) == len(FOCUS_ROUTE_FIELDS):
            if parts[-1] == "0":
                parts[-1] = ""
            route = dict(zip(FOCUS_ROUTE_FIELDS, parts))
    return place_id if isinstance(place_id, str) else "", route


def _encode_component(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _route_names(route: Mapping[str, Any]) -> tuple[str, ...]:
    provider, account, conversation, thread = (route.get(key) for key in FOCUS_ROUTE_FIELDS)
    if not (isinstance(provider, str) and provider and isinstance(account, str)
            and isinstance(conversation, str) and conversation and isinstance(thread, (str, type(None)))):
        return ()
    path = "/".join(map(_encode_component, (provider, account, conversation)))
    name = f"conversation:{path}"
    if thread:
        return name, f"thread:{path}/{_encode_component(thread)}"
    return (name,)


def _place_name(place_id: str) -> str:
    return f"place:{_encode_component(place_id)}"


CARRIED_NAMES = "situation_names"  # a helper's situation names, written by value by the work that scheduled it


def carried_names(task: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The situation names a helper carries (``CARRIED_NAMES``, under ``metadata`` or at the top level), else ``None``."""
    value = _mapping(task.get("metadata")).get(CARRIED_NAMES, task.get(CARRIED_NAMES))
    return tuple(name for name in value if isinstance(name, str)) if isinstance(value, list) else None


def carried_room(task: Mapping[str, Any]) -> str | None:
    """The room a helper's carried names name, which its memory view reads; ``None`` when it carries none."""
    return next((name.removeprefix("room:") for name in carried_names(task) or () if name.startswith("room:")), None)


def helper_names(ctx: Any, root: Path) -> tuple[str, ...]:
    """The names a helper this work starts carries: the ones this work holds heads by (its room as the memory view
    resolves it for its role) but ``WAKE``, which stays with the wake; a helper passes on its carried names, ``WAKE``
    never among them, and one that carries none passes none."""
    from ouroboros.memory_inventory import membership_facts
    from ouroboros.memory_view import view_role, view_room_id

    task = {"metadata": _mapping(getattr(ctx, "task_metadata", None)), "task_contract": getattr(ctx, "task_contract", None)}
    role = view_role(task)
    if role in DELEGATED_FOCUS_ROLES or carried_names(task) is not None:
        names = carried_names(task) or ()
    else:
        room = view_room_id(task, ctx, root, project_chat_ids=membership_facts(root).project_chat_ids,
                            presence=role == "presence")
        names = situation_names(task, room)
    return tuple(name for name in names if name != WAKE)


def situation_names(task: Mapping[str, Any], room_id: str) -> tuple[str, ...]:
    """The names from host facts, given the room already resolved by the memory view; a helper's are those it carries.
    An own-clock wake (``is_own_clock_wake``, read at the top level or under ``metadata`` as the view role is) also
    has ``WAKE``; a focus wake, a Main turn and the work a wake starts have their own names only. Carried names keep
    ``WAKE`` only for an own-clock wake (ordinary admission keeps what a task carries; ``helper_names``' rule)."""
    own_clock = is_own_clock_wake({**task, **_mapping(task.get("metadata"))})
    if (carried := carried_names(task)) is not None:
        return tuple(name for name in carried if own_clock or name != WAKE)
    place_id, route = _place_facts(task)
    names = (f"room:{room_id}",)
    if place_id:
        names += (_place_name(place_id),)
    return names + _route_names(route) + ((WAKE,) if own_clock else ())


@dataclass(frozen=True)
class SituationReceipt:
    root: Path
    room_labels: RoomLabelResolver
    places: Mapping[str, Any] | None
    witnessed: tuple[str, ...]
    line: str

    def matching(self, names: list[str]) -> tuple[set[str], set[str]]:
        """The proved matches and the names whose evidence could not be read."""
        wanted, unverified = set(names), set()
        known = {WAKE, f"room:{WEB_UI_CHAT_ID}", *(f"room:{chat}" for chat in self.room_labels.project_chat_ids)}
        known.update(self.witnessed)
        if self.places is None:
            unverified.update(name for name in wanted if name.startswith("place:"))
        else:
            known.update(_place_name(place.place_id) for place in self.places.values())
        unresolved = {name for name in wanted - known if name.startswith(("room:", "conversation:", "thread:"))}
        if unresolved:
            try:
                if dialogue_store.present(self.root):
                    coverage = dialogue_store.range_coverage(self.root, "chat")
                    skip = self.room_labels.project_chat_ids | {WEB_UI_CHAT_ID, HIDDEN_CHAT_ID}
                    for chat, route in dialogue_store.transport_witnesses(self.root, skip_chat_ids=skip):
                        route_names = _route_names(route)
                        if route_names and _derives(route, chat):
                            known.add(f"room:{chat}")
                            known.update(route_names)
                    if not coverage["imported"] or coverage["gaps"]:
                        unverified.update(unresolved - known)
            except (OSError, StoreError):
                unverified.update(unresolved - known)
        return wanted & known, unverified


def receipt_situation(ctx: Any, root: Path, *, claims_written: bool = False) -> SituationReceipt | None:
    """Resolve the same room and role as the memory view, for a knowledge tool receipt."""
    from ouroboros.memory_view import view_role, view_room_id

    metadata = _mapping(getattr(ctx, "task_metadata", None))
    task = {"metadata": metadata, "task_contract": getattr(ctx, "task_contract", None)}
    role = view_role(task)
    integrating = role not in DELEGATED_FOCUS_ROLES
    if not integrating and not claims_written:
        return None

    from ouroboros.places import PlaceError, place_rows  # D03->D20 is lazy-only

    try:
        places = place_rows(root)
    except (PlaceError, OSError):
        places = None
    labels = RoomLabelResolver(root, places=places or {})
    room = view_room_id(task, ctx, root, project_chat_ids=labels.project_chat_ids, presence=role == "presence")
    place_id, route = _place_facts(task)
    focus = _mapping(metadata.get("focus"))
    source_room, _ = labels._chat_id({"chat_id": focus.get("room_chat_id", room if role == "presence" else None)})
    route_names = _route_names(route)
    witnessed = (f"room:{source_room}", *route_names) if source_room is not None and route_names and _derives(route, source_room) else ()
    if not integrating:
        return SituationReceipt(root, labels, places, witnessed, "")

    names = situation_names(task, room)
    room_label = labels.label({"chat_id": room, "transport": route}).removesuffix(f" [chat_id={room}]")
    descriptions = {names[0]: room_label, WAKE: "a wake my own clock started"}
    place = (places or {}).get(place_id)
    carried_place = _mapping(_mapping(metadata.get("presence")).get("place"))
    if place_id:
        descriptions[_place_name(place_id)] = place.label if place else carried_place.get("label", "")
    conversation = _mapping(route.get("conversation"))
    if route_names:
        descriptions[route_names[0]] = next((conversation[key] for key in ("title", "name", "full_name", "username")
                                             if isinstance(conversation.get(key), str) and conversation[key]), "")
    rendered = []
    for name in names:
        label = " ".join(str(descriptions.get(name) or "").split())
        rendered.append(f"{name} ({label})" if label else name)
    return SituationReceipt(root, labels, places, witnessed, "This situation's names: " + ", ".join(rendered))
