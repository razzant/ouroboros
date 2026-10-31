"""Canonical presentation membership for indexed history, evidence and memory readers.

A row is written once, to one address. ``row_rooms`` says every room that shows it: the room
its work lives in by the room rule, the room it was written to for host and private rows,
and, for the owner's direct Reply to a Project message, that Project as well. Readers address
their candidates through the dialogue store (``gateway.history_paging.room_lens``,
``chat_chain.room_rows``: a superset, including ``appearance_keys`` for rows written elsewhere);
this rule decides each one.
"""
from __future__ import annotations

from typing import Any, Callable, FrozenSet, List, Mapping, Optional

# ``chat_references.reference_from_row`` stamps what decides a Reply target's room, read at
# Reply time from the target row itself: its task's room ("work": a later binding moves the
# link with the task), the room it was written in ("room"), or none (host receipts and
# refusals of an act in Main, private Presence contact).
TARGET_WORK, TARGET_ROOM, TARGET_NONE = "work", "room", "none"
# A saved target whose stored bytes cannot be read: its room is unknown, never guessed. The Reply
# stays Main's; the Projects it could join find ``GAP_KEY`` keys and disclose the gap. One proven
# absent (both streams readable, no row at its address) is no room's. Either is a fact of the store
# it was read under: the rooms the saved reference names stay candidates, decided again on a read
# once a stream's end moved (an import or a repair may bring the target).
TARGET_UNKNOWN, TARGET_ABSENT, GAP_KEY = "unknown", "absent", "gap:"


def _ref_chat(ref: Mapping[str, Any]) -> int:
    try:
        return int(ref.get("chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def reply_target_home(target: Mapping[str, Any]) -> str:
    """What decides the room of one Reply target row (``TARGET_*``); never its words."""
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES

    kind = target.get("type")
    if private_main_row(dict(target)) or kind in (MAIN_PINNED_ROW_TYPES | ORIGIN_ADDRESSED_NOTICE_TYPES) - {
            "project_completion_summary", "relay_reply"}:
        return TARGET_NONE
    return TARGET_WORK if target.get("direction") != "in" and target.get("task_id") else TARGET_ROOM


# (root, address) -> ((home, task, chat), ends): a found target row is immutable, read once (ends None);
# a target not found holds while the dialogue streams' ends it was read under stand.
_SAVED_TARGETS: dict = {}
_SAVED_TARGETS_MAX = 4096


def _stream_ends(root: Any) -> tuple:
    """The dialogue streams' ends (``dialogue_store.stream_end``, the readers' change counter): an append and
    a newly planned segment move them. The archive import stores rows under unchanged ends, but only into
    intervals planned before it, and while one is pending ``resolve_row`` answers importing, so no not-found
    decision is kept. No code rewrites the rows or skipped lines of an imported interval in place; were it
    done by hand, this process would see it after the next append or a restart."""
    from ouroboros import dialogue_store as ds

    return tuple(ds.stream_end(root, stream) for stream in ds.STREAMS)


def _saved_target(rooms: Any, ref: Mapping[str, Any]) -> tuple:
    """``(home, task, chat)`` of a Reply target saved before ``target_home``, read from the target row
    itself at its exact address (chat, then progress, as ``chat_references.resolve_reply`` does).
    One proven absent is ``TARGET_ABSENT``; one whose bytes cannot be read (a recorded unreadable
    interval, an ambiguous address, another row of its chat and time beside unreadable lines, as
    ``chat_references.resolve_history_row`` holds) is ``TARGET_UNKNOWN``: both with the saved
    reference's task and chat, the rooms it could join, and read again once the store moved. An
    unreadable or importing store raises its typed error, the reader's gap, never a guess."""
    address = str(ref.get("address") or "")
    if address.startswith("task:"):  # a work card names its work; it has no row of its own
        return TARGET_WORK, address[5:], 0
    root = getattr(rooms, "root", None)
    if not address or root is None:
        return TARGET_NONE, "", 0
    key = (str(root), address)
    held = _SAVED_TARGETS.get(key)
    if held is not None and held[1] is None:
        return held[0]
    ends = _stream_ends(root)  # read first: a row stored during the read below moves them past it
    if held is not None and held[1] == ends:
        return held[0]
    from ouroboros import dialogue_positions as dp
    from ouroboros.chat_chain import _row_chat_id, resolve_row

    target, statuses = None, set()
    for stream in ("chat", "progress"):
        target, found = resolve_row(root, address, stream=stream)
        if target:
            break
        status = found.get("status")
        statuses.add("row_unreadable" if status == "row_mismatch" and dp.skipped_between(root, stream) else status)
    if target:
        decision, ends = (reply_target_home(target), str(target.get("task_id") or ""), _row_chat_id(target)), None
    else:
        decision = (TARGET_UNKNOWN if statuses & {"row_unreadable", "row_ambiguous"} else TARGET_ABSENT,
                    str(ref.get("task_id") or ""), _ref_chat(ref))
    _SAVED_TARGETS.pop(key, None)
    _SAVED_TARGETS[key] = (decision, ends)
    while len(_SAVED_TARGETS) > _SAVED_TARGETS_MAX:
        _SAVED_TARGETS.pop(next(iter(_SAVED_TARGETS)))
    return decision


def _could(task: str, chat: int, written: int, project_chat_ids, rooms: Callable[..., int]) -> tuple:
    """The Projects other than the written room a Reply whose target is not found could join as
    either home: its saved task's room now and the chat its saved reference names."""
    return tuple(sorted(({int(rooms(task) or 0) if task else 0, chat} & set(project_chat_ids)) - {written}))


def unknown_target_rooms(row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> FrozenSet[int]:
    """The Projects an owner Reply could be a member of while its saved target cannot be read
    (``TARGET_UNKNOWN``): a read of theirs may lack it and says so; empty for every other row."""
    ref = row.get("reply_to") if row.get("direction") == "in" else None
    if not isinstance(ref, dict) or ref.get("unresolved") or ref.get("target_home") is not None:
        return frozenset()
    home, task, chat = _saved_target(rooms, ref)
    if home != TARGET_UNKNOWN:
        return frozenset()
    try:
        written = int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        written = 1
    return frozenset(_could(task, chat, written, project_chat_ids, rooms))


def reply_target_room(row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> int:
    """The Project of an owner row's direct Reply target, 0 when none: one hop, never its ancestry.

    The reference is the host's, resolved when the row was accepted; one saved before
    ``target_home`` existed is decided by its actual target row (``_saved_target``)."""
    ref = row.get("reply_to") if row.get("direction") == "in" else None
    if not isinstance(ref, dict) or ref.get("unresolved"):
        return 0
    chat, task = _ref_chat(ref), str(ref.get("task_id") or "")
    home = ref.get("target_home")
    if home is None:
        home, task, chat = _saved_target(rooms, ref)
    if home == TARGET_WORK and task:
        room = int(rooms(task) or 0)
        return room if room in project_chat_ids else 0
    return chat if home == TARGET_ROOM and chat in project_chat_ids else 0


def row_rooms(entry_chat: int, row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> FrozenSet[int]:
    """Every room the room rule puts one stored row in, as chat ids (binding-origin copies apart).

    ``rooms`` is the room rule (``projects_registry.RoomResolver``), the one the live bus addresses
    by, so a task's rows replay where its frames went: a later binding moves ordinary task speech.
    Host rows about an act keep their issuing room; work speech written elsewhere keeps that room
    too and joins its work's room."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import (MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES,
                                            WORK_SPEECH_ELSEWHERE_TYPES, bound_room_chat)

    if is_a2a_chat_id(entry_chat):
        return frozenset()
    if private_main_row(dict(row)):
        return frozenset({1})
    if row_provenance(row) == "place":
        return frozenset() if entry_chat == 1 else frozenset({entry_chat})
    kind = row.get("type")
    speech = kind in WORK_SPEECH_ELSEWHERE_TYPES
    bound = 0 if speech or kind in ORIGIN_ADDRESSED_NOTICE_TYPES else bound_room_chat(rooms, row)
    lifecycle = kind in MAIN_PINNED_ROW_TYPES
    found = set()
    if not lifecycle and (bound or entry_chat) in project_chat_ids:
        found.add(bound or entry_chat)
    if not bound and entry_chat not in project_chat_ids and entry_chat != 1:
        found.add(entry_chat)
    if entry_chat != HIDDEN_CHAT_ID and entry_chat not in project_chat_ids and (lifecycle or not bound):
        found.add(1)
    extra = bound_room_chat(rooms, row) if speech else reply_target_room(row, project_chat_ids, rooms)
    if extra in project_chat_ids:
        found.add(extra)
    return frozenset(found)


def appearance_keys(row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> List[str]:
    """The ``dialogue_store`` appearance index keys of a row some Project shows outside its written
    room: ``room:<chat>`` where that room is fixed when the row is written, ``task:<id>`` for a Reply
    whose target follows its task, so a later binding reaches it through the Project's bound tasks.
    A candidate superset; ``row_rooms`` decides. A Reply whose saved target is not found takes the keys
    it could have as either home, so a read after an import or a repair finds and decides it; one whose
    target is ``TARGET_UNKNOWN`` takes them behind ``GAP_KEY`` too: the Projects it could join disclose
    it while it stays unknown (``room_appearances.unknown_targets`` decides it again)."""
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES, bound_room_chat

    try:
        written = int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        written = 1
    keys: set = set()
    if row.get("type") in WORK_SPEECH_ELSEWHERE_TYPES and not private_main_row(dict(row)):
        keys.add(bound_room_chat(rooms, row))
    elif isinstance(row.get("reply_to"), dict) and row.get("direction") == "in" and not row["reply_to"].get("unresolved"):
        ref = row["reply_to"]
        keys.add(reply_target_room(row, project_chat_ids, rooms))
        home, task, chat = ((ref["target_home"], str(ref.get("task_id") or ""), 0) if "target_home" in ref
                            else _saved_target(rooms, ref))
        if home == TARGET_WORK and task:
            keys.add(f"task:{task}")
        elif home in (TARGET_UNKNOWN, TARGET_ABSENT):  # the keys it would have as either home stay candidates
            could = [*([f"task:{task}"] if task else ()),
                     *(f"room:{room}" for room in _could(task, chat, written, project_chat_ids, rooms))]
            keys.update([*could, *(GAP_KEY + key for key in could if home == TARGET_UNKNOWN)])
    return sorted(f"room:{key}" if isinstance(key, int) else key for key in keys
                  if not isinstance(key, int) or (key in project_chat_ids and key != written))


def binding_origin_allowed(entry_chat: int, row: Mapping[str, Any]) -> bool:
    """Whether a Project may also show this row as the owner message its binding names: never
    A2A traffic, private contact, a place's row or a host row pinned to Main."""
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES

    return not (is_a2a_chat_id(entry_chat) or private_main_row(dict(row)) or row_provenance(row) == "place"
                or row.get("type") in MAIN_PINNED_ROW_TYPES)


def room_membership(chat_id: int, project_chat_ids: set, source_refs: list,
                    rooms: Callable[..., int]):
    """Canonical room membership shared by history, exact-row and evidence readers.

    ``gateway.history_paging.room_lens`` addresses the Project admissions in the dialogue
    store (its candidate superset); change both together. A Project also shows the owner
    message a binding names (its origin), kept by the room it was written in.
    Presentation-only hiding and cross-room question pointers belong to the UI caller.
    """
    from ouroboros.project_dialogue import entry_matches_source_ref

    def matches(entry_chat: int, entry: Optional[dict] = None) -> bool:
        row = entry if isinstance(entry, dict) else {}
        if chat_id in row_rooms(entry_chat, row, project_chat_ids, rooms):
            return True
        return bool(chat_id in project_chat_ids and source_refs and binding_origin_allowed(entry_chat, row)
                    and entry_matches_source_ref(row, source_refs))

    return matches
