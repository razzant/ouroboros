"""Writer facts and the catch-up job of the dialogue store's derived appearance index.

A chat row written to one chat that a Project also shows (``room_membership.row_rooms``: work
speech written to Main or a relay's source room, private Presence contact, the owner's direct
Reply to a Project message) is found by that Project's readers through ``dialogue_store``'s
``room_appearances`` keys, never by walking another room. A new row takes its keys in its own
append transaction (``supervisor.message_ingress``). The maintenance job ``room_appearances``
derives rows at or above the store's mark: rows stored before the index, and rows an older body
appended. Once, it also derives again, below the mark it began under, Main's rows an older rule
kept there alone: private Presence contact and the owner's direct Replies whose target that rule
could not follow (``_contact_replies``), and so again the rows an older body wrote after this one
(``_owe_uncovered``) and the planned rows another body imported (``owe_planned``, ``imported_here``).
It is idempotent and touches nothing but the index and the newest-message times of the Projects it finds a
message in (``room_messages.shown``): no row, unread revision, notification or delivery.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import room_messages

log = logging.getLogger(__name__)

_BATCH = 500  # rows per read and per index transaction


def _rule(root: Any) -> Tuple[frozenset, Any]:
    from ouroboros.projects_registry import RoomResolver, reserved_project_chat_ids

    # Certification persists: unavailable authority must leave derivation owed, not certify an empty room set.
    return frozenset(reserved_project_chat_ids(root, strict=True)), RoomResolver(root, strict=True)


def _candidate(row: Mapping[str, Any]) -> bool:
    """Whether a row can be shown outside its written chat: everything else needs no rule read."""
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES

    return row.get("type") in WORK_SPEECH_ELSEWHERE_TYPES or private_main_row(dict(row)) or (
        row.get("direction") == "in" and isinstance(row.get("reply_to"), dict))


def write_facts(root: Any, row: Mapping[str, Any]) -> Tuple[Optional[List[str]], List[int]]:
    """``(index keys, other Projects that show it)`` for one chat row about to be written.

    Keys are ``None`` when the rule cannot be read now: the row remains at or above the
    store's mark for the job, never recorded as needing none."""
    if not _candidate(row):
        return [], []
    from ouroboros.room_membership import appearance_keys, row_rooms

    try:
        projects, rooms = _rule(root)
        written = int(row.get("chat_id", 1))
        shown = sorted(chat for chat in row_rooms(written, row, projects, rooms) if chat in projects and chat != written)
        return appearance_keys(row, projects, rooms), shown
    except Exception:
        log.debug("appearance facts unavailable for a %s row; the job derives them", row.get("type"), exc_info=True)
        return None, []


def frame_rooms(root: Any, row: Optional[Mapping[str, Any]], *, stamped_only: bool = False) -> Dict[str, List[int]]:
    """``{"also_rooms": [...]}``: the other Projects a saved row's live frame also reaches, the rooms its
    revision stamp names or else the rule's answer now (never with ``stamped_only``); ``{}`` when none.
    A frame is not a second row."""
    if not row:
        return {}
    stamp = row.get("room_revisions")
    try:
        written = int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        return {}
    if isinstance(stamp, dict):
        shown = sorted(int(chat) for chat in stamp if str(chat).lstrip("-").isdigit() and int(chat) != written)
    else:
        shown = [] if stamped_only else write_facts(root, row)[1]
    return {"also_rooms": shown} if shown else {}


def room_project(root: Any, chat_id: Any, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    """The registry row reserving one Project room (any lifecycle), or ``None``; one registry parse serves a whole read.
    ``strict`` lets an unreadable registry raise (a history window that cannot place a row fails), else it is ``None``."""
    from ouroboros.projects_registry import project_of_room

    try:
        return project_of_room(root, int(chat_id))
    except Exception:
        if strict:
            raise
        log.debug("Project room %r unreadable", chat_id, exc_info=True)
        return None


def frame_project(root: Any, row: Optional[Mapping[str, Any]], shown: Optional[List[int]] = None) -> Dict[str, Any]:
    """``{project_id, project_name}``: the active Project whose work a row outside that room speaks of (a notice to
    Main, a relay's answer, private Presence contact), read from the one room the row already stands in (``shown``,
    else the row's room stamp: a row written before rooms were stamped names none), so the row carries that Project's
    reference (DESIGN "References and actions"). ``{}`` for every other row. A projection, never a stored copy: a
    renamed Project reads its new name."""
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES
    from ouroboros.project_facts import project_display_name
    from ouroboros.projects_registry import PROJECT_ACTIVE

    if not row or not (row.get("type") in WORK_SPEECH_ELSEWHERE_TYPES or private_main_row(dict(row))):
        return {}
    try:
        rooms = frame_rooms(root, row, stamped_only=True).get("also_rooms", []) if shown is None else shown
        project = room_project(root, rooms[0]) if len(rooms) == 1 else None
    except Exception:
        log.debug("Project of a %s row unreadable", row.get("type"), exc_info=True)
        return {}
    if not project or project.get("lifecycle") != PROJECT_ACTIVE:
        return {}
    return {"project_id": str(project["id"]), "project_name": project_display_name(project.get("name"), project["id"])}


def mark(root: Any) -> Optional[int]:
    """The stored coordinate below which every chat row is derived; ``None`` before the first pass."""
    from ouroboros import dialogue_store as ds

    value = ds.meta(root, ds.APPEARANCES_MARK)
    return value if type(value) is int else None


def _older_imports(root: Any) -> List[Tuple[int, int, int]]:
    """``(segment id, base, end)`` of chat segments this body planned and an older body imported:
    ``done`` while still awaiting. Empty awaiting (the steady state) reads one meta row."""
    from ouroboros import dialogue_store as ds

    with ds.transaction(root, write=False) as conn:
        awaiting = ds._meta(conn, ds.APPEARANCES_AWAITING) or []
        if not awaiting:
            return []
        records = conn.execute(f"SELECT id, base, size FROM import_ledger WHERE id IN ({','.join('?' * len(awaiting))}) "
                               "AND state='done'", [int(entry) for entry in awaiting]).fetchall()
    return [(int(record["id"]), int(record["base"]), int(record["base"]) + int(record["size"])) for record in records]


def owed(root: Any) -> bool:
    """Whether rows wait for derivation: at or above the mark (none derived yet, a pass cut short, an
    older body's appends), in a segment an older body imported below it, the contact Replies' pass, or
    below the mark but not covered by this body's rule (an older body wrote them after this one)."""
    from ouroboros import dialogue_store as ds

    if not ds.present(root):
        return False
    below, done, covered = (mark(root), ds.meta(root, ds.APPEARANCES_CONTACT_REPLIES),
                            ds.meta(root, ds.APPEARANCES_COVERED))
    return (below is None or below < ds.stream_end(root, "chat") or bool(_older_imports(root))
            or not isinstance(done, dict) or done["next"] < done["upper"] or type(covered) is not int
            or covered < below)


def pending_below(root: Any, upper: int, lower: int = 0) -> bool:
    """Whether a read of ``[lower, upper)`` may lack a row shown here but written elsewhere: the derivation
    has not passed it (a store from before the index, an older body's appends or imports), or it reaches
    the one-time pass's unfinished ``[next, upper)`` or the rows from where this body's rule stopped covering
    up to the mark; before that pass is named, every row below the mark is unknown. Meta reads in the steady
    state; unreadable is pending."""
    from ouroboros import dialogue_store as ds

    try:
        below, contact, covered = (mark(root), ds.meta(root, ds.APPEARANCES_CONTACT_REPLIES),
                                   ds.meta(root, ds.APPEARANCES_COVERED))
        start, stop = (contact["next"], contact["upper"]) if isinstance(contact, dict) else (0, below or 0)
        covered = covered if type(covered) is int else stop
        return (below is None or below < upper or any(base < upper for _id, base, _end in _older_imports(root))
                or any(first < last and lower < last and first < upper
                       for first, last in ((start, stop), (covered, below))))
    except Exception:
        log.debug("appearance derivation facts unreadable; reported pending", exc_info=True)
        return True


def unknown_targets(root: Any, keys: Iterable[str], lower: int, upper: int, *, chat: Optional[int] = None) -> bool:
    """Whether a Reply in ``[lower, upper)`` could join a room with these appearance keys (``chat``, or any
    room they reach) but its saved target cannot be read (``room_membership.TARGET_UNKNOWN``): one seek per
    key and gap prefix (this rule's, and the one an older body may have erased) finds the rows derived so,
    and each is decided again, so a target an import or a repair made readable closes its gap at once; an
    unreadable fact is a gap too."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_membership import GAP_KEYS, unknown_target_rooms

    wanted = [gap + key for gap in GAP_KEYS for key in keys]
    if not wanted or upper <= lower:
        return False
    try:
        with ds.transaction(root, write=False) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='room_appearances'").fetchone():
                return False
            seqs = json.loads(conn.execute(
                "SELECT json_group_array(DISTINCT seq) FROM room_appearances WHERE stream='chat' AND key IN "
                "(SELECT value FROM json_each(?)) AND seq>=? AND seq<?",
                (json.dumps(wanted), int(lower), int(upper))).fetchone()[0])
        if not seqs:
            return False
        rows, (projects, rooms) = ds.rows_at(root, "chat", seqs), _rule(root)
        for seq in seqs:
            if seq not in rows:  # a row the index names that the store does not give: unknown
                return True
            could = unknown_target_rooms(rows[seq].row, projects, rooms)
            if chat in could if chat is not None else could:
                return True
        return False
    except Exception:
        log.debug("appearance gap facts unreadable; reported as a gap", exc_info=True)
        return True


# The kinds of row a Project read through the index may lack (``read_gaps``): the history window
# names them per stream (``chat_appearances_deriving``), memory and evidence coverage by kind.
APPEARANCES_DERIVING, REPLY_TARGET_UNREADABLE = "appearances_deriving", "reply_target_unreadable"
GAP_DETAILS = {APPEARANCES_DERIVING: "Rows this Project shows but that were written elsewhere are still being indexed",
               REPLY_TARGET_UNREADABLE: "A Reply whose saved target cannot be read may belong to this Project"}


def read_gaps(root: Any, chat: int, keys: Iterable[str], lower: int, upper: int) -> set:
    """What a read of one Project's chat rows in ``[lower, upper)`` through its appearance ``keys`` may lack:
    ``APPEARANCES_DERIVING`` while the derivation has not passed ``upper``, ``REPLY_TARGET_UNREADABLE``
    while a Reply there could join it but its saved target cannot be read."""
    return ({APPEARANCES_DERIVING} if pending_below(root, upper, lower) else set()) | (
        {REPLY_TARGET_UNREADABLE} if unknown_targets(root, keys, lower, upper, chat=chat) else set())


def due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """After the archive import (positions settle first); a failed pass waits for the child's other work."""
    from ouroboros import dialogue_store as ds

    if row.get("state") == "failed":
        return child
    try:
        ds.require_imported(root, "chat")
        return owed(root)
    except Exception:
        log.debug("appearance index facts unreadable; the next look asks again", exc_info=True)
        return False


def run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """Derive the rows from the mark to the stream's end, one index transaction a batch.

    The mark moves only from the value this pass read (an append moves it only from exactly its
    own start), so rows appended meanwhile stay above it for the next pass."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_membership import appearance_keys

    projects, rooms = _rule(root)
    start = mark(root)
    with ds.transaction(root, write=True) as conn:  # rows below the mark this pass began under had the older rule
        conn.execute(ds.APPEARANCES_DDL)
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (ds.APPEARANCES_MARK, json.dumps(0)))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                     (ds.APPEARANCES_CONTACT_REPLIES, json.dumps({"next": 0, "upper": start or 0}, sort_keys=True)))
        contact = ds._meta(conn, ds.APPEARANCES_CONTACT_REPLIES)
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",  # the pass owes what lies below
                     (ds.APPEARANCES_COVERED, json.dumps(contact["upper"] if isinstance(contact, dict) else start or 0)))
    start = mark(root) or 0
    end, items, batch = ds.stream_end(root, "chat"), 0, []

    def commit(upto: int) -> bool:
        nonlocal start, items
        keys = [(key, "chat", entry.seq) for entry in batch for key in (
            appearance_keys(entry.row, projects, rooms) if _candidate(entry.row) else ())]
        dated = room_messages.shown(batch, projects, rooms)
        with ds.transaction(root, write=True) as conn:
            conn.execute(ds.APPEARANCES_DDL)
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, ?, ?)", keys)
            room_messages.note(conn, dated)
            moved = conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                                 (json.dumps(upto), ds.APPEARANCES_MARK, json.dumps(start))).rowcount
            if moved:  # derived by this body's rule
                conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                             (json.dumps(upto), ds.APPEARANCES_COVERED, json.dumps(start)))
        items += len(keys)
        start = upto if moved else start
        batch.clear()
        return bool(moved)

    for entry in ds.iter_rows(root, "chat", lower=start, upper=end, batch=_BATCH):
        batch.append(entry)
        if len(batch) == _BATCH:
            if not commit(entry.seq + entry.length) or should_stop():
                return {"state": "pending", "items": items}
    if not commit(end):
        return {"state": "pending", "items": items}
    for segment, base, upper in _older_imports(root):  # exactly the rows an older body imported below the mark
        keys, dated = [], []
        for entry in ds.iter_rows(root, "chat", lower=base, upper=upper, batch=_BATCH):
            keys.extend((key, "chat", entry.seq) for key in (appearance_keys(entry.row, projects, rooms)
                                                             if _candidate(entry.row) else ()))
            dated.extend(room_messages.shown([entry], projects, rooms))
        with ds.transaction(root, write=True) as conn:
            conn.execute(ds.APPEARANCES_DDL)
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, ?, ?)", keys)
            room_messages.note(conn, dated)
            ds._set_meta(conn, ds.APPEARANCES_AWAITING,
                         [entry for entry in ds._meta(conn, ds.APPEARANCES_AWAITING) or [] if entry != segment])
        items += len(keys)
        if should_stop():
            return {"state": "pending", "items": items}
    for _round in range(2):  # the pass's interval, then the rows an older body wrote past this body's coverage
        _owe_uncovered(root)
        replies, finished = _contact_replies(root, projects, rooms, should_stop)
        items += replies
        if not finished:
            return {"state": "pending", "items": items}
    items += _rederive_gaps(root, projects, rooms)
    return {"state": "done" if not owed(root) else "pending", "items": items}


def _owe_uncovered(root: Any) -> None:
    """Owe a finished contact pass the rows an older body wrote after this one: its appends, imports and job moved
    the mark but never this body's coverage (``dialogue_store.APPEARANCES_COVERED``), so exactly ``[covered, mark)``
    becomes the pass. An unfinished pass first ends its own interval; the two never merge. One meta transaction."""
    from ouroboros import dialogue_store as ds

    with ds.transaction(root, write=True) as conn:
        below, covered, state = (ds._meta(conn, key) for key in (
            ds.APPEARANCES_MARK, ds.APPEARANCES_COVERED, ds.APPEARANCES_CONTACT_REPLIES))
        if (type(below) is not int or type(covered) is not int or not isinstance(state, dict) or covered >= below
                or state["next"] < state["upper"]):
            return
        _set_pass(conn, covered, below)
        conn.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(below), ds.APPEARANCES_COVERED))


def _set_pass(conn: Any, first: int, last: int) -> None:
    """The contact pass now owes ``[first, last)``, in the exact bytes its cursor compares."""
    from ouroboros import dialogue_store as ds

    conn.execute("UPDATE meta SET value=? WHERE key=?",
                 (json.dumps({"next": first, "upper": last}, sort_keys=True), ds.APPEARANCES_CONTACT_REPLIES))


def owe_restored_targets(conn: Any) -> None:
    """New archive coverage in either stream can restore a target previously proven absent: a plan after the
    store's cutover fact. Before it every target not found reads as importing (``dialogue_store.require_imported``),
    never absent, so the first cutover's plan owes nothing here.
    Reuse the resumable Reply pass over the existing chat prefix; imports retire only their own
    interval, leaving these older dependants owed. Ordinary appends never restart this pass."""
    from ouroboros import dialogue_store as ds

    state = ds._meta(conn, ds.APPEARANCES_CONTACT_REPLIES)
    if isinstance(state, dict) and ds._meta(conn, "cutover") is not None:
        end = int(conn.execute("SELECT next_seq FROM streams WHERE stream='chat'").fetchone()[0])
        _set_pass(conn, 0, max(end, state["upper"]))


def owe_planned(conn: Any, start: int, end: int) -> None:
    """In the planning transaction that moved the mark and this body's coverage over ``[start, end)``: a planned
    segment is no evidence its rows took this body's rule, whoever imports it (an older body retires its awaiting
    entry as it imports, deriving by its own rule), so the pass owes them (an unfinished pass reaches them from its
    own ``next``) until this body's import derives them (``imported_here``)."""
    from ouroboros import dialogue_store as ds

    state = ds._meta(conn, ds.APPEARANCES_CONTACT_REPLIES)
    if isinstance(state, dict) and start < end:
        _set_pass(conn, state["next"] if state["next"] < state["upper"] else start, max(end, state["upper"]))


def imported_here(conn: Any, lower: int, upper: int) -> None:
    """In this body's import transaction, after its rule derived the segment ``[lower, upper)``: the rows leave the
    pass from its top, as imports go newest first. A segment below a top still owed (rows another body imported, or
    one still to come) leaves the pass as it is: what it owes is exact or a bounded superset, never short."""
    from ouroboros import dialogue_store as ds

    state = ds._meta(conn, ds.APPEARANCES_CONTACT_REPLIES)
    if isinstance(state, dict) and lower < state["upper"] <= upper:
        _set_pass(conn, state["next"], max(state["next"], lower))


def _contact_replies(root: Any, projects: frozenset, rooms: Any, should_stop: Callable[[], bool]) -> Tuple[int, bool]:
    """``(keys, finished)`` of the one-time/restoration pass below its ``upper``, one index transaction
    a batch. Private Presence contact (``private_main_row``: my notice, my
    question, its answer) and an owner's direct Reply saved with a "none" stamp or with none at all take
    the keys ``room_membership.appearance_keys`` gives them now (a Reply's decision reads its exact target;
    any other stays as it was). The cursor moves only from the value this pass read, so a resumed or a
    second pass adds nothing twice and a pass cut short resumes where it stopped."""
    from ouroboros import dialogue_store as ds
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.room_membership import TARGET_NONE, appearance_keys

    def older(row: Mapping[str, Any]) -> bool:  # what the older rule left in Main alone
        ref = row.get("reply_to")
        return private_main_row(dict(row)) or (row.get("direction") == "in" and isinstance(ref, dict)
                                               and ref.get("target_home", TARGET_NONE) == TARGET_NONE)

    state, items = ds.meta(root, ds.APPEARANCES_CONTACT_REPLIES), 0
    while isinstance(state, dict) and state["next"] < state["upper"]:
        with ds.transaction(root, write=False) as conn:
            page = ds._page(conn, "WHERE stream='chat' "
                            "AND seq>=? AND seq<? ORDER BY seq LIMIT ?", (state["next"], state["upper"], _BATCH))
        entries = [ds._row(record) for record in ds._decoded(page)]
        upto = entries[-1].seq + 1 if len(entries) == _BATCH else state["upper"]
        keys = [(key, "chat", entry.seq) for entry in entries if older(entry.row)
                for key in appearance_keys(entry.row, projects, rooms)]
        dated = room_messages.shown([entry for entry in entries if older(entry.row)], projects, rooms)
        moved = {**state, "next": upto}
        with ds.transaction(root, write=True) as conn:
            if not conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?", (
                    json.dumps(moved, sort_keys=True), ds.APPEARANCES_CONTACT_REPLIES, json.dumps(state, sort_keys=True))).rowcount:
                return items, False  # another pass moved it: the next look continues from there
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, ?, ?)", keys)
            room_messages.note(conn, dated)
        items, state = items + len(keys), moved
        if should_stop() and state["next"] < state["upper"]:
            return items, False
    return items, True


def _rederive_gaps(root: Any, projects: frozenset, rooms: Any) -> int:
    """Derive again the rows that hold gap keys of either prefix (two ranges of the index, never history): a
    row whose saved target an import or a repair made readable since loses both and takes the keys of its home
    now; one still unknown takes back the prefix an older body's job erased. A reader decides each such row
    again anyway (``unknown_targets``); this keeps the index honest."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_membership import GAP_KEYS, appearance_keys

    ranges = [(gap, gap[:-1] + chr(ord(gap[-1]) + 1)) for gap in GAP_KEYS]  # every key behind a prefix, one range
    with ds.transaction(root, write=False) as conn:
        seqs = sorted({seq for gaps in ranges for seq in json.loads(conn.execute(
            "SELECT json_group_array(DISTINCT seq) FROM room_appearances WHERE key>=? AND key<? AND stream='chat'",
            gaps).fetchone()[0])})
    rows = ds.rows_at(root, "chat", seqs) if seqs else {}
    fresh = {seq: appearance_keys(entry.row, projects, rooms) for seq, entry in rows.items()}
    closed = {seq for seq, keys in fresh.items() if not any(key.startswith(GAP_KEYS) for key in keys)}
    if fresh:
        dated = room_messages.shown([rows[seq] for seq in closed], projects, rooms)  # a target read since: its room
        with ds.transaction(root, write=True) as conn:
            conn.executemany("DELETE FROM room_appearances WHERE stream='chat' AND seq=? AND key>=? AND key<?",
                             [(seq, *gaps) for seq in closed for gaps in ranges])
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, 'chat', ?)",
                             [(key, seq) for seq, keys in fresh.items() for key in keys])
            room_messages.note(conn, dated)
    return len(closed)
