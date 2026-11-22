"""Canonical presentation membership for indexed history, evidence and memory readers.

A row is written once, to one address. ``row_rooms`` says every room that shows it: the room
its work lives in by the room rule, the room it was written to for host rows and Main for private
Presence contact (which also joins its work's Project), and, for the owner's direct Reply to a
Project message, that Project as well. Readers address
their candidates through the dialogue store (``gateway.history_paging.room_lens``,
``chat_chain.room_rows``: a superset, including ``appearance_keys`` for rows written elsewhere);
this rule decides each one.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, FrozenSet, List, Mapping, Optional

# ``chat_references.reference_from_row`` stamps what decides a Reply target's room, read at
# Reply time from the target row itself: its task's room ("work": a later binding moves the
# link with the task), the room it was written in ("room"), or none (host receipts and
# refusals of an act in Main). Private Presence contact was "none" before it joined its
# work's Project; ``reply_target`` reads such a saved stamp again from its exact target.
TARGET_WORK, TARGET_ROOM, TARGET_NONE = "work", "room", "none"
# The reference roles a private contact's saved stamp takes (``reference_from_row``): my words (its notice
# or question) and the answer row the host records for that question (a system reference, as a host
# receipt's is: its exact target decides, never the role). Role, not kind: the memory inventory keeps a
# reference's role (``memory_inventory._REPLY_FIELDS``), so every reader decides alike.
_CONTACT_ROLES = frozenset({"assistant", "system"})
# A saved target whose stored bytes cannot be read: its room is unknown, never guessed. The Reply
# stays Main's; the Projects it could join find ``GAP_KEY`` keys and disclose the gap. One proven
# absent (both streams readable, no row at its address) is no room's. Either is a fact of the store
# it was read under: the rooms the saved reference names stay candidates, decided again on a read
# once a stream's end moved (an import or a repair may bring the target) or a text was revised.
TARGET_UNKNOWN, TARGET_ABSENT, GAP_KEY = "unknown", "absent", "gap:"
# The same fact as this rule decides it, beside ``GAP_KEY`` (which an older body reads and, by its own rule, may
# erase): outside that key's range, so only this rule's catch-up closes it, both keys together.
CURRENT_GAP_KEY = "gap2:"
GAP_KEYS = (CURRENT_GAP_KEY, GAP_KEY)


def _ref_chat(ref: Mapping[str, Any]) -> int:
    try:
        return int(ref.get("chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def reply_target_home(target: Mapping[str, Any]) -> str:
    """What decides the room of one Reply target row (``TARGET_*``); never its words. Private
    Presence contact is its work's speech: a Reply to it follows that work, one hop."""
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES

    if target.get("type") in (MAIN_PINNED_ROW_TYPES | ORIGIN_ADDRESSED_NOTICE_TYPES) - {
            "project_completion_summary", "relay_reply"}:
        return TARGET_NONE
    return TARGET_WORK if target.get("direction") != "in" and target.get("task_id") else TARGET_ROOM


# (root, address) -> (target only, dialogue identity, revision, ends). An unlocated target
# carries no task or room of its own; each reference supplies its own fallback.
_SAVED_TARGETS: dict = {}
_SAVED_TARGETS_MAX = 4096
_CAPTURED_TARGETS: ContextVar[Optional[dict]] = ContextVar("captured_reply_targets", default=None)


def _store_identity(root: Any) -> Optional[tuple]:
    from ouroboros import dialogue_store as ds, store

    return store.file_identity(ds.path(root))


def capture_reply_targets(rooms: Any, read: Callable[[], Any]) -> Any:
    """Batch a pure whole-view membership pass, not a transaction around its Python work.

    Take version marks lazily so a pass with no unstamped target costs nothing.
    A store identity or chat text revision that moved during the pass makes its answers
    stale: read again once, ordinary live resolution. A row appended meanwhile restarts
    no grouping (this pass's own list of rows is fixed), and a target that stays
    unresolved is decided again, under fresh marks, by the next read. The closure
    returns its own facts, never writes them.
    """
    capture = {"rooms": rooms}
    token = _CAPTURED_TARGETS.set(capture)
    try:
        result = read()
    finally:
        _CAPTURED_TARGETS.reset(token)
    if "version" in capture:
        root = rooms.root
        if capture["version"] != (_store_identity(root), _text_revision(root)):
            return read()
    return result


def _text_revision(root: Any) -> int:
    """The chat text revision ``dialogue_store.revise_text`` counts: a revised row keeps its coordinate and
    answers another address, so under unchanged stream ends a target found may answer no more and one not
    found may answer now. A progress row's revision is not counted; no caller revises one."""
    from ouroboros import dialogue_store as ds

    return int(ds.meta(root, "chat_text_revision") or 0)


def _stream_ends(root: Any) -> tuple:
    """The dialogue streams' ends (``dialogue_store.stream_end``, the readers' change counter): an append and
    a newly planned segment move them. The archive import stores rows under unchanged ends, but only into
    intervals planned before it: while one is pending a row not found answers importing (``resolve_row``), and
    ``_target_facts`` looks once more after a miss, so no absence outlives an import. An ambiguous address
    answers before that check and stays unknown: no import makes it exact. Apart from a text revision
    (``_text_revision``) no code rewrites the rows or skipped lines of an imported interval in place; were it
    done by hand, this process would see it after the next append, text revision or restart."""
    from ouroboros import dialogue_store as ds

    return tuple(ds.stream_end(root, stream) for stream in ds.STREAMS)


def _saved_target(rooms: Any, ref: Mapping[str, Any]) -> tuple:
    """``(home, task, chat)`` of a Reply target saved before ``target_home``, read from the target row
    itself at its exact address (chat, then progress, as ``chat_references.resolve_reply`` does), as
    that address answers now: no link is kept apart from it, so a target whose text was revised is
    not found until its text is back, and its Reply is shown only where it was written (owner decision
    2026-11-11).
    One proven absent is ``TARGET_ABSENT``; one whose bytes cannot be read (a recorded unreadable
    interval, an ambiguous address, another row of its chat and time beside unreadable lines, as
    ``chat_references.resolve_history_row`` holds) is ``TARGET_UNKNOWN``: both with the saved
    reference's task and chat, the rooms it could join, and read again once the store moved. An
    unreadable or importing store raises its typed error, the reader's gap, never a guess."""
    return _target_facts(rooms, ref)[:3]


def _target_facts(rooms: Any, ref: Mapping[str, Any]) -> tuple:
    """``_saved_target``'s answer and whether the row found there is private Presence contact."""
    address = str(ref.get("address") or "")
    if address.startswith("task:"):  # a work card names its work; it has no row of its own
        return TARGET_WORK, address[5:], 0, False
    root = getattr(rooms, "root", None)
    if not address or root is None:
        return TARGET_NONE, "", 0, False
    key = (str(root), address)
    capture = _CAPTURED_TARGETS.get()
    if capture is not None and capture["rooms"] is rooms:
        if "version" not in capture:
            capture["version"] = _store_identity(root), _text_revision(root)
        identity, revision = capture["version"]
    else:
        capture = None
        identity, revision = _store_identity(root), _text_revision(root)
    held = _SAVED_TARGETS.get(key) if identity is not None else None
    if held is not None and held[1:] == (identity, revision, None):
        return held[0]
    if capture is None:
        ends = _stream_ends(root)
    else:
        if "ends" not in capture:
            capture["ends"] = _stream_ends(root)
        ends = capture["ends"]
    # A cached miss is a fact of the stream ends it was decided under: an import between two of
    # one pass's lookups moves them and stores the target at its exact address, so read them
    # again now before reusing one -- no absence outlives an import the pass itself may meet.
    # A found target never pays for that, and a lookup outside a pass read its pair above, as before.
    if held is not None and held[3] is not None:
        fresh = ends if capture is None else _stream_ends(root)
        if held[1:] == (identity, revision, fresh):
            return held[0][0], str(ref.get("task_id") or ""), _ref_chat(ref), False
    from ouroboros import dialogue_positions as dp
    from ouroboros.chat_chain import _row_chat_id, resolve_row
    from ouroboros.presence_owner_contact import private_main_row

    # A miss looks once more. A stream whose miss was not an ambiguous address answered imported in the first
    # look (``resolve_row``), so the second reads the rows an import stored meanwhile, under unchanged marks.
    for _look in range(2):
        target, statuses = None, set()
        for stream in ("chat", "progress"):
            target, found = resolve_row(root, address, stream=stream)
            if target:
                break
            status = found.get("status")
            statuses.add("row_unreadable" if status == "row_mismatch" and dp.skipped_between(root, stream) else status)
        if target:
            break
    if target:
        decision, ends = (reply_target_home(target), str(target.get("task_id") or ""), _row_chat_id(target),
                          private_main_row(dict(target))), None
    else:
        decision = (TARGET_UNKNOWN if statuses & {"row_unreadable", "row_ambiguous"} else TARGET_ABSENT,
                    "", 0, False)
    if identity is not None:
        _SAVED_TARGETS[key] = (decision, identity, revision, ends)
        while len(_SAVED_TARGETS) > _SAVED_TARGETS_MAX:
            _SAVED_TARGETS.pop(next(iter(_SAVED_TARGETS)), None)
    return decision if target else (decision[0], str(ref.get("task_id") or ""), _ref_chat(ref), False)


def reply_target(rooms: Any, ref: Mapping[str, Any]) -> tuple:
    """``(home, task, chat)`` of one direct Reply's target: the one decision membership, gap disclosure
    and the appearance index (its derivation and the job's one-time pass) all read. The host's stamp,
    else the target row itself (``_saved_target``). A "none" stamp a private contact's reference could
    carry (a row in Main with its task: my notice or question, or its answer) is read again at its exact
    address: the target proven private Presence contact follows its work, as a Reply saved now does
    (owner choice, every such Reply); one unreadable is ``TARGET_UNKNOWN``; any other "none" stays none."""
    home = ref.get("target_home")
    if home is None:
        return _saved_target(rooms, ref)
    task, chat = str(ref.get("task_id") or ""), _ref_chat(ref)
    if home == TARGET_NONE and task and chat == 1 and ref.get("role") in _CONTACT_ROLES:
        found, found_task, found_chat, private = _target_facts(rooms, ref)
        if private:
            return TARGET_WORK, found_task, found_chat
        if found == TARGET_UNKNOWN:
            return TARGET_UNKNOWN, task, chat
    return home, task, chat


def _could(task: str, chat: int, written: int, project_chat_ids, rooms: Callable[..., int]) -> tuple:
    """The Projects other than the written room a Reply whose target is not found could join as
    either home: its saved task's room now and the chat its saved reference names."""
    return tuple(sorted(({int(rooms(task) or 0) if task else 0, chat} & set(project_chat_ids)) - {written}))


def unknown_target_rooms(row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> FrozenSet[int]:
    """The Projects an owner Reply could be a member of while its saved target cannot be read
    (``TARGET_UNKNOWN``): a read of theirs may lack it and says so; empty for every other row."""
    ref = row.get("reply_to") if row.get("direction") == "in" else None
    if not isinstance(ref, dict) or ref.get("unresolved") or "telegram_room_chat_id" in ref:  # decided: no gap
        return frozenset()
    home, task, chat = reply_target(rooms, ref)
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
    ``target_home`` existed, or stamped as a private contact once was, is decided by its actual
    target row (``reply_target``). The bundled Telegram bridge's Reply keeps the room its turn was
    given at acceptance (``telegram_room_chat_id``; 0: its own conversation's), never decided again."""
    ref = row.get("reply_to") if row.get("direction") == "in" else None
    if not isinstance(ref, dict) or ref.get("unresolved"):
        return 0
    if "telegram_room_chat_id" in ref:  # the room the host decided once for the bundled bridge's Reply: never again
        room = ref["telegram_room_chat_id"]
        return room if type(room) is int and room > 0 and room in project_chat_ids else 0
    home, task, chat = reply_target(rooms, ref)
    if home == TARGET_WORK and task:
        room = int(rooms(task) or 0)
        return room if room in project_chat_ids else 0
    return chat if home == TARGET_ROOM and chat in project_chat_ids else 0


def stamped_room_echo(row: Mapping[str, Any], entry_chat: int) -> bool:
    """Whether a stored row is an accepted web echo whose persisted ``kind: "room_echo"`` stamp limits
    its display, replay included, to the room it was written in (CLIENT_CONTRACT §5; the live twin is
    ``chat_activity.mainThreadAccepts``). The producer writes the stamp only on an accepted inbound web
    row outside Main, so Main is never a stamped row's *guessed* room: a reservation lost after the
    acceptance leaves the row in its written room instead of resurfacing it in Main. The stamp grants
    no routing authority — unstamped legacy and external-transport rows keep their implicit Main
    mirror, a row written to Main stays Main's, and a Reply-target appearance (``reply_target_room``)
    is untouched."""
    return (entry_chat != 1 and row.get("kind") == "room_echo"
            and row.get("direction") == "in" and row.get("source") == "web")


def row_rooms(entry_chat: int, row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> FrozenSet[int]:
    """Every room the room rule puts one stored row in, as chat ids (binding-origin copies apart).

    ``rooms`` is the room rule (``projects_registry.RoomResolver``), the one the live bus addresses
    by, so a task's rows replay where its frames went: a later binding moves ordinary task speech.
    Host rows about an act keep their issuing room; work speech written elsewhere keeps that room
    too and joins its work's room. An accepted web echo keeps its persisted display stamp
    (``stamped_room_echo``): a reservation lost after the acceptance never guesses it into Main.
    Private Presence contact (its notice, question and their answers)
    is Main's and its work's Project's, never a place room's: one row, wherever it is shown. A place's
    rows stay its room's; the host's Done row of work it started also stands in the Project that work
    lives in, as every root's does."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import (MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES,
                                            WORK_SPEECH_ELSEWHERE_TYPES, bound_room_chat)

    if is_a2a_chat_id(entry_chat):
        return frozenset()
    from ouroboros.chat_references import telegram_chat_row

    if row.get("direction") == "in" and telegram_chat_row(row.get("source"), entry_chat, row.get("client_message_id")):
        # The bridge's own chat id is a Telegram address, never a room's: its owner rows are Main's and that
        # conversation's, never a Project bearing the same number by that number alone; a Reply joins the room the
        # host decided for its turn, even that Project's (an older row's target there is this conversation's).
        ref = row.get("reply_to")
        extra = reply_target_room(row, project_chat_ids, rooms)
        decided = isinstance(ref, dict) and "telegram_room_chat_id" in ref
        return frozenset({1, *({entry_chat} if entry_chat not in project_chat_ids else ()),
                          *({extra} if extra and (decided or extra != entry_chat) else ())})
    if private_main_row(dict(row)):
        return frozenset({1, bound_room_chat(rooms, row)} & {1, *project_chat_ids})
    kind = row.get("type")
    if entry_chat == 1 and kind == "task_summary" and row.get("schedule_report_only") is True:
        return frozenset({1})  # a schedule's local terminal card is not a Place reply
    if row_provenance(row) == "place":  # a place's rows stay its room's; its work's Done row is its Project's too
        work = bound_room_chat(rooms, row) if kind == "task_summary" else 0
        return frozenset(({entry_chat} if entry_chat != 1 else set()) | ({work} if work in project_chat_ids else set()))
    speech = kind in WORK_SPEECH_ELSEWHERE_TYPES
    if row.get("direction") == "out" and row.get("logical_room_chat_id") == 1 and not kind:
        task = str(row.get("task_id") or "")
        if task not in getattr(rooms, "bindings", {}):
            return frozenset({1, *({entry_chat} if entry_chat not in project_chat_ids else ())})
    bound = 0 if speech or kind in ORIGIN_ADDRESSED_NOTICE_TYPES else bound_room_chat(rooms, row)
    lifecycle = kind in MAIN_PINNED_ROW_TYPES
    found = set()
    if not lifecycle and (bound or entry_chat) in project_chat_ids:
        found.add(bound or entry_chat)
    if not bound and entry_chat not in project_chat_ids and entry_chat != 1:
        found.add(entry_chat)
    if (entry_chat != HIDDEN_CHAT_ID and entry_chat not in project_chat_ids and (lifecycle or not bound)
            and not stamped_room_echo(row, entry_chat)):  # a stamped echo keeps its written address, replay included
        found.add(1)
    extra = bound_room_chat(rooms, row) if speech else reply_target_room(row, project_chat_ids, rooms)
    if extra in project_chat_ids:
        found.add(extra)
    return frozenset(found)


def appearance_keys(row: Mapping[str, Any], project_chat_ids, rooms: Callable[..., int]) -> List[str]:
    """The ``dialogue_store`` appearance index keys of a row some Project shows outside its written
    room: ``room:<chat>`` where that room is fixed when the row is written (work speech and private
    contact: their work's room then), ``task:<id>`` for a Reply whose target follows its task, so a
    later binding reaches it through the Project's bound tasks (and their own rows by their task ids).
    A candidate superset; ``row_rooms`` decides. A Reply whose saved target is not found takes the keys
    it could have as either home, so a read after an import or a repair finds and decides it; one whose
    target is ``TARGET_UNKNOWN`` takes them behind ``GAP_KEYS`` too: the Projects it could join disclose
    it while it stays unknown (``room_appearances.unknown_targets`` decides it again)."""
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES, bound_room_chat

    try:
        written = int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        written = 1
    if written != 1 and row.get("direction") == "out" and row.get("logical_room_chat_id") == 1:
        return ["room:1"]  # membership still follows a later Project binding, if any
    keys: set = set()
    if row.get("type") in WORK_SPEECH_ELSEWHERE_TYPES or private_main_row(dict(row)):
        keys.add(bound_room_chat(rooms, row))
    elif isinstance(row.get("reply_to"), dict) and row.get("direction") == "in" and not row["reply_to"].get("unresolved"):
        keys.add(reply_target_room(row, project_chat_ids, rooms))
        home, task, chat = reply_target(rooms, row["reply_to"])
        if home == TARGET_WORK and task:
            keys.add(f"task:{task}")
        elif home in (TARGET_UNKNOWN, TARGET_ABSENT):  # the keys it would have as either home stay candidates
            could = [*([f"task:{task}"] if task else ()),
                     *(f"room:{room}" for room in _could(task, chat, written, project_chat_ids, rooms))]
            keys.update([*could, *(gap + key for gap in GAP_KEYS for key in could if home == TARGET_UNKNOWN)])
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
