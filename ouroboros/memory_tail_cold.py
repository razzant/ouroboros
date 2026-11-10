"""The cold set: the stored tail rows a capture may leave unread, and the marks it stands on.

A capture that holds every row after the frontier (``memory_inventory.open_rows_by_room``) reads the stored facts of the
whole tail to find the open ones. A row may stay unread only while nothing it gives that capture can change: every
input of its rooms is written in the row itself or covered by a mark, and no reader takes it for anything else. A pass
over the whole tail stores those rows (``tail_cold``, positions beside ``tail_rows``) and the marks it read them under
(``state.tail_cold``) in one index transaction; a later capture compares the marks in the same read that takes the
seals and the rows (``chronicle_store._read_snapshot``) and, on any difference, reads the whole tail as before and
composes the set anew. No reader changes: every row the set leaves out is one no reader of the capture uses.

Marks, all equal or the whole tail:

- the stored tail's source (``memory_inventory._tail_source``: dialogue store, frontier, chat text revision, facts
  version), as the index's own ``state.tail`` names it, complete;
- the registry by value: the very projects, bindings, origin messages and Project rooms the capture's rule reads;
- the task store's lineage writes (``store_hot_facts.lineage_writes``: its triggers count every change or removal of a
  task's parent, root, role or admitted chat in that writer's own transaction, whichever body wrote it);
- the running code (an update composes anew);
- nothing journaled after the set's sequence returns a row to reading: no decision that rejected a page (its seals
  go), no page by a writer whose terminal row the set left unread (``memory_sealing`` links it below the page).

Never cold: a row open in some room; each room's newest row (it says whether the room is a place); every row of a
place room (its earlier deliveries of the same words, who of ours spoke there); a quiet finish (it considers the words
before it); an owner's Reply (its target may appear or move, ``room_membership.reply_target``); a terminal row of a
task that wrote a page record; a row whose answer rested on a task the store did not hold or could not read
(``RoomResolver.settled``). A row the set does not name, appended or inserted since, is read. A room that became a
place since the set was composed makes the capture read the whole tail (``short``).

The set is a projection: a reset of the chronicle index takes it with its marks (``_RESET``), and the first whole pass
after an update, a rebuild or any change composes it again; nothing to migrate or fill. A doubt is a whole pass: a
stale set costs one, a needed row left unread would cost a word.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional

STANDS, GATE, HELD = "tail_cold", "tail_cold_gate", "tail_cold_held"  # a capture's facts of the set (``capture_memo``)
MEMO = (STANDS, GATE, HELD)
MARKS = "tail_cold"  # the set's marks, in the index's ``state``
_SINCE = "SELECT body FROM records WHERE sequence>? AND kind IN ('decision','page') ORDER BY sequence"


class Gate(NamedTuple):
    """What a capture of the whole tail reads before its index transaction: the membership facts its rule decides rooms
    by (one object) and the marks a stored set must name to stand."""
    facts: Any
    marks: Dict[str, Any]


def gate(store: Any) -> Optional[Gate]:
    """The marks outside the index, read once for this capture; ``None`` when one cannot be read: then nothing stands
    and nothing is composed, and the capture reads the whole tail."""
    from ouroboros.memory_inventory import _membership_facts
    from ouroboros.projects_registry import project_thread_chat_ids
    from ouroboros.transcript_prefix import _running_code

    try:
        facts = _membership_facts(store.data_root)
        if facts.lineage is None:
            return None
        rooms = facts.rooms
        rooms._projects = project_thread_chat_ids(store.data_root)  # read here once: the mark and the rule read the same
        registry = [sorted(facts.project_chat_ids), facts.bindings, rooms.bindings, sorted(rooms._projects),
                    sorted([list(key), sorted(chats)] for key, chats in facts.projects_by_source_key.items())]
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return Gate(facts, {"registry": digest, "lineage": facts.lineage, "code": _running_code()})
    except Exception:  # a mark not read decides nothing: the whole tail
        return None


def stands(db: Any, gate_: Gate) -> Optional[Dict[str, Any]]:
    """The stored set's marks when this read transaction may leave its rows unread, else ``None``: they are the gate's,
    over the stored tail as it stands complete now, and nothing journaled after the set's sequence returns a row."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory_sealing import _terminal_writer

    held, tail = (ChronicleStore._state(db, key) for key in (MARKS, "tail"))
    if not (isinstance(held, dict) and isinstance(tail, dict) and tail.get("complete")
            and held.get("marks") == gate_.marks and held.get("source") == tail.get("source")):
        return None
    for (body,) in db.execute(_SINCE, (held["sequence"],)):
        record = json.loads(body)
        if record.get("kind") == "decision" and not record.get("accepted"):  # its page's seals are gone
            return None
        writer = _terminal_writer(record)
        if writer and db.execute("SELECT 1 FROM tail_cold WHERE task=? LIMIT 1", (writer,)).fetchone():
            return None
    return held


def memo(gate_: Optional[Gate], held: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A capture's facts of the set: its membership facts (the rule's and the marks' one object), the gate, and whether
    and on which marks it stands; nothing when no gate was read."""
    return {} if gate_ is None else {"membership": gate_.facts, GATE: gate_, STANDS: held is not None, HELD: held or {}}


class Watch:
    """The room rule's task answers, watched row by row: ``unsure`` says whether one rested on a task the store did not
    hold or could not read (``RoomResolver.settled``); such a task may appear and move the row, so it is never cold."""
    __slots__ = ("rooms", "root", "_unsure")

    def __init__(self, rooms: Any) -> None:
        self.rooms, self.root, self._unsure = rooms, rooms.root, False

    def __call__(self, task_id: Any, parent_task_id: Any = "", root_task_id: Any = "") -> int:
        answer = self.rooms(task_id, parent_task_id, root_task_id)
        if not self.rooms.settled(task_id, parent_task_id, root_task_id):
            self._unsure = True
        return answer

    def unsure(self) -> bool:
        """Whether a row judged since the last ask rested on such a task; asking starts the next row."""
        unsure, self._unsure = self._unsure, False
        return unsure


def short(derived: Optional[Mapping[str, Any]], tail: Any, places: Iterable[str]) -> bool:
    """On a capture standing on the set: whether a room is a place now that was none when the set was composed (its
    earlier sealed rows may be among those left unread); then it reads the whole tail again. A capture whose rows were
    read live (the stored facts stand for another source) left none unread. Either stands no more."""
    if not (derived and derived.get(STANDS)) or tail.complete and set(places) <= set(derived[HELD].get("places") or ()):
        return False
    derived[STANDS] = False  # type: ignore[index]
    return tail.complete


def settle(store: Any, derived: Optional[Mapping[str, Any]], tail: Any, found: Any) -> None:
    """Count what this capture's pass read (``rows_cold``: the rows the set left unread; ``tail_whole``: a pass over
    the whole tail) and, after a whole pass of a capture that read the marks over a complete stored tail, store what no
    reader used as the set, with those marks, in one index transaction. A busy index or a stored tail replaced
    meanwhile stores nothing: the next whole pass does."""
    from ouroboros.context_timing import counted

    if derived and derived.get(STANDS):
        counted("rows_cold", int(derived[HELD].get("rows") or 0))
        return
    counted("tail_whole")
    snapshot = store._pinned()
    if derived and derived.get(GATE) and tail.complete and snapshot is not None and not snapshot.proposed:
        _compose(store, snapshot, derived[GATE], tail, found)


def _compose(store: Any, snapshot: Any, gate_: Gate, tail: Any, found: Any) -> None:
    import sqlite3

    from ouroboros.memory_sealing import _terminal_writer

    writers = {_terminal_writer(snapshot.body(row)) for row in snapshot.records if row[2] == "page"} - {""}
    # A terminal row's task is kept with it: a page its task writes later links it (``stands``).
    cold = [[pos, (meta.get("task_id") or None) if terminal is not None else None]
            for pos, _address, meta, terminal in tail.rows if pos not in found.kept and not (
                meta.get("quiet_finish") is True or meta.get("direction") == "in" and isinstance(meta.get("reply_to"), dict)
                or terminal is not None and str(meta.get("task_id") or "") in writers)]
    source = (snapshot.state("tail") or {}).get("source")
    marks = {"marks": gate_.marks, "source": source, "sequence": snapshot.records[-1][0] if snapshot.records else 0,
             "places": sorted(found.places), "rows": len(cold)}
    try:
        with store._publishing(0) as db:
            held = store._state(db, "tail")
            if not isinstance(held, dict) or held.get("source") != source:
                return
            db.execute("DELETE FROM tail_cold")
            db.execute("INSERT INTO tail_cold SELECT json_extract(value,'$[0]'), json_extract(value,'$[1]') "
                       "FROM json_each(?)", (json.dumps(cold),))
            store._set_state(db, MARKS, marks)
            db.commit()
    except (TimeoutError, sqlite3.Error):
        return
