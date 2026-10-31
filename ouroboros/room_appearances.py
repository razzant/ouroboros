"""Writer facts and the catch-up job of the dialogue store's derived appearance index.

A chat row written to one chat that a Project also shows (``room_membership.row_rooms``: work
speech written to Main or a relay's source room, the owner's direct Reply to a Project message)
is found by that Project's readers through ``dialogue_store``'s ``room_appearances`` keys, never by
walking another room. A new row takes its keys in its own append transaction
(``supervisor.message_ingress``). The maintenance job ``room_appearances`` derives rows at or above
the store's mark: rows stored before the index, and rows an older body appended. It is idempotent
and touches nothing but the index: no row, revision, notification or delivery.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

_BATCH = 500  # rows per read and per index transaction


def _rule(root: Any) -> Tuple[frozenset, Any]:
    from ouroboros.projects_registry import RoomResolver, reserved_project_chat_ids

    # Certification persists: unavailable authority must leave derivation owed, not certify an empty room set.
    return frozenset(reserved_project_chat_ids(root, strict=True)), RoomResolver(root, strict=True)


def _candidate(row: Mapping[str, Any]) -> bool:
    """Whether a row can be shown outside its written chat: everything else needs no rule read."""
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES

    return row.get("type") in WORK_SPEECH_ELSEWHERE_TYPES or (
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


def frame_rooms(root: Any, row: Optional[Mapping[str, Any]]) -> Dict[str, List[int]]:
    """``{"also_rooms": [...]}``: the other Projects a saved row's live frame also reaches, the rooms its
    revision stamp names or else the rule's answer now; ``{}`` when none. A frame is not a second row."""
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
        shown = write_facts(root, row)[1]
    return {"also_rooms": shown} if shown else {}


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
    older body's appends) or in a segment an older body imported below it."""
    from ouroboros import dialogue_store as ds

    if not ds.present(root):
        return False
    below = mark(root)
    return below is None or below < ds.stream_end(root, "chat") or bool(_older_imports(root))


def pending_below(root: Any, upper: int) -> bool:
    """Whether a read up to ``upper`` may lack a row shown here but written elsewhere: the derivation
    has not passed it (a store from before the index, an older body's appends or imports). Meta reads
    in the steady state; unreadable is pending."""
    try:
        below = mark(root)
        return below is None or below < upper or any(base < upper for _id, base, _end in _older_imports(root))
    except Exception:
        log.debug("appearance derivation facts unreadable; reported pending", exc_info=True)
        return True


def unknown_targets(root: Any, keys: Iterable[str], lower: int, upper: int, *, chat: Optional[int] = None) -> bool:
    """Whether a Reply in ``[lower, upper)`` could join a room with these appearance keys (``chat``, or any
    room they reach) but its saved target cannot be read (``room_membership.TARGET_UNKNOWN``): one seek per
    key finds the rows derived so, and each is decided again, so a target an import or a repair made
    readable closes its gap at once; an unreadable fact is a gap too."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_membership import GAP_KEY, unknown_target_rooms

    wanted = [GAP_KEY + key for key in keys]
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
    return ({APPEARANCES_DERIVING} if pending_below(root, upper) else set()) | (
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
    if start is None:
        with ds.transaction(root, write=True) as conn:
            conn.execute(ds.APPEARANCES_DDL)
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (ds.APPEARANCES_MARK, json.dumps(0)))
        start = mark(root) or 0
    end, items, batch = ds.stream_end(root, "chat"), 0, []

    def commit(upto: int) -> bool:
        nonlocal start, items
        keys = [(key, "chat", entry.seq) for entry in batch for key in (
            appearance_keys(entry.row, projects, rooms) if _candidate(entry.row) else ())]
        with ds.transaction(root, write=True) as conn:
            conn.execute(ds.APPEARANCES_DDL)
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, ?, ?)", keys)
            moved = conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                                 (json.dumps(upto), ds.APPEARANCES_MARK, json.dumps(start))).rowcount
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
        keys = [(key, "chat", entry.seq) for entry in ds.iter_rows(root, "chat", lower=base, upper=upper, batch=_BATCH)
                for key in (appearance_keys(entry.row, projects, rooms) if _candidate(entry.row) else ())]
        with ds.transaction(root, write=True) as conn:
            conn.execute(ds.APPEARANCES_DDL)
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, ?, ?)", keys)
            ds._set_meta(conn, ds.APPEARANCES_AWAITING,
                         [entry for entry in ds._meta(conn, ds.APPEARANCES_AWAITING) or [] if entry != segment])
        items += len(keys)
        if should_stop():
            return {"state": "pending", "items": items}
    items += _rederive_gaps(root, projects, rooms)
    return {"state": "done" if not owed(root) else "pending", "items": items}


def _rederive_gaps(root: Any, projects: frozenset, rooms: Any) -> int:
    """Derive again the rows that hold gap keys (one range of the index, never history): a row whose saved
    target an import or a repair made readable since loses them and takes the keys of its home now. A
    reader decides each such row again anyway (``unknown_targets``); this keeps the index honest."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_membership import GAP_KEY, appearance_keys

    gaps = (GAP_KEY, GAP_KEY[:-1] + chr(ord(GAP_KEY[-1]) + 1))  # every key behind the prefix, one index range
    with ds.transaction(root, write=False) as conn:
        seqs = json.loads(conn.execute("SELECT json_group_array(DISTINCT seq) FROM room_appearances WHERE key>=? AND key<? "
                                       "AND stream='chat'", gaps).fetchone()[0])
    rows = ds.rows_at(root, "chat", seqs) if seqs else {}
    fresh = {seq: appearance_keys(entry.row, projects, rooms) for seq, entry in rows.items()}
    closed = {seq: keys for seq, keys in fresh.items() if not any(key.startswith(GAP_KEY) for key in keys)}
    if closed:
        with ds.transaction(root, write=True) as conn:
            conn.executemany("DELETE FROM room_appearances WHERE stream='chat' AND seq=? AND key>=? AND key<?",
                             [(seq, *gaps) for seq in closed])
            conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, 'chat', ?)",
                             [(key, seq) for seq, keys in closed.items() for key in keys])
    return len(closed)
