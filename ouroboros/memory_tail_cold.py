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
- the code this process loaded (``CODE``: an update composes anew);
- nothing journaled after the set's sequence returns a row to reading: no decision that rejected a page (its seals
  go), no page by a writer whose terminal row the set left unread (``memory_sealing`` links it below the page).

Never cold: a row open in some room; each room's newest row (it says whether the room is a place); every row of a
Project whose newest row is a place's (its rows come from many chats); a quiet finish (it considers the words before
it); an owner's Reply (its target may appear or move, ``room_membership.reply_target``); a terminal row of a task that
wrote a page record; a row whose answer rested on a task the store did not hold or could not read
(``RoomResolver.settled``). A row the set does not name, appended or inserted since, is read. The sealed rows of any
other place room (its earlier deliveries of the same words, who of ours spoke there) are read for the rooms a reader
shows, when it asks (``older_of``): the set keeps each row's room key, the chat ``chat_chain._row_chat_id`` reads, as
the room rule reads it, and no other room holds a row of that chat. A Project that became a place since the set was
composed makes the capture read the whole tail (``again``).

One capture, one state: the marks and the facts the rule decides rooms by are read together (the gate's one object),
and a task's lineage, read later as the rule asks, is checked against its mark after the pass and after the readers
are done (``again``, ``moved``); every later read of the set's rows checks the set and the stored tail in its own
transaction (``_read``). Where any moved, the capture reads the whole tail again over the facts as they are then
(``Moved``, ``memory_inventory.over_one_state``), as when no gate was read.

The set is a projection: a reset of the chronicle index takes it with its marks (``_RESET``), and the first whole pass
after an update, a rebuild or any change composes it again; nothing to migrate or fill. A doubt is a whole pass: a
stale set costs one, a needed row left unread would cost a word.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional

from ouroboros.transcript_prefix import _running_code

STANDS, GATE, HELD = "tail_cold", "tail_cold_gate", "tail_cold_held"  # a capture's facts of the set (``capture_memo``)
MEMO = (STANDS, GATE, HELD)
MARKS = "tail_cold"  # the set's marks, in the index's ``state``
_SINCE = "SELECT body FROM records WHERE sequence>? AND kind IN ('decision','page') ORDER BY sequence"
# One page of the set's rows a selection of positions names (its parameters first), stored facts as
# ``chronicle_store._tail_rows`` reads them; the selections: the newest rows of one chat before a position (a bounded
# read of its index, whatever its history), the rows of some chats with given words.
_FACTS = ("SELECT json_group_array(json_array(t.pos,t.seq,t.row_sha256,json(t.legacy),json(t.meta),t.terminal)), "
          "max(c.pos) FROM (SELECT pos FROM ({}) WHERE pos>? ORDER BY pos LIMIT ?) c JOIN tail_rows t ON t.pos=c.pos")
_OLDER = "SELECT pos FROM (SELECT pos FROM tail_cold WHERE chat=? AND pos<? ORDER BY pos DESC LIMIT ?)"
_END = 1 << 62  # past every position: a chat's newest rows
_COPIES = ("SELECT pos FROM tail_cold WHERE (chat, digest) IN "
           "(SELECT json_extract(value,'$[0]'), json_extract(value,'$[1]') FROM json_each(?))")
_PAGE = 2_000
# The code this process runs, named as it loads the set (with the chronicle, at start): an update restarts every process,
# and a commit checked out later, before the restart that loads it, is not this process's code.
CODE = _running_code()


class Gate(NamedTuple):
    """What a capture of the whole tail reads before its index transaction: the membership facts its rule decides rooms
    by (one object), the marks a stored set must name to stand, and the registry files' identities as the facts were
    read (``moved``)."""
    facts: Any
    marks: Dict[str, Any]
    files: tuple = ()


class Moved(Exception):
    """What a capture's set stood on moved before its readers were done: the capture reads the whole tail again."""


def gate(store: Any) -> Optional[Gate]:
    """The marks outside the index, read once for this capture; ``None`` when one cannot be read: then nothing stands
    and nothing is composed, and the capture reads the whole tail."""
    from ouroboros.memory_inventory import _membership_facts

    try:
        files = _registry_files(store.data_root)
        facts = _membership_facts(store.data_root)
        if facts.lineage is None:
            return None
        rooms = facts.rooms  # its Project rooms were read once with the facts: the mark and the rule read the same
        registry = [sorted(facts.project_chat_ids), facts.bindings, rooms.bindings, sorted(rooms._projects),
                    sorted([list(key), sorted(chats)] for key, chats in facts.projects_by_source_key.items())]
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return Gate(facts, {"registry": digest, "lineage": facts.lineage, "code": CODE}, files)
    except Exception:  # a mark not read decides nothing: the whole tail
        return None


def _registry_files(root: Any) -> tuple:
    from ouroboros import utils
    from ouroboros.projects_registry import _bindings_path, _registry_path, _registry_witness_path

    stamps = []
    for path in (_registry_path(root), _registry_witness_path(root), _bindings_path(root)):
        try:
            stamps.append(utils.file_identity(path.stat()))
        except OSError:
            stamps.append(None)
    return tuple(stamps)


def moved(store: Any, gate_: Gate) -> bool:
    """Whether what ``gate_`` read for a capture changed since: a task's lineage written (one counter read), or the
    registry's facts (read again only when one of its files was written since)."""
    from ouroboros.store_hot_facts import lineage_writes

    if lineage_writes(store.data_root) != gate_.marks["lineage"]:
        return True
    if _registry_files(store.data_root) == gate_.files:
        return False
    now = gate(store)
    return now is None or now.marks["registry"] != gate_.marks["registry"]


def forgo(derived: Dict[str, Any]) -> None:
    """A capture whose set moved under it stands on nothing from here: it composes nothing, and its rule answers each
    task again from the lineage as it is now (this process's memory of it dropped where the counter moved)."""
    from ouroboros.projects_registry import forget_lineage
    from ouroboros.store_hot_facts import lineage_writes

    derived.update({STANDS: False, HELD: {}})
    derived.pop(GATE, None)
    facts = derived.get("membership")
    if facts is not None:
        forget_lineage(facts.rooms.root, lineage_writes(facts.rooms.root), facts.rooms._projects)
        facts.rooms.answers.clear()


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


def again(store: Any, derived: Optional[Dict[str, Any]], tail: Any, places: Iterable[str]) -> bool:
    """After the pass of a capture standing on the set: whether it reads the whole tail instead. A Project that is a
    place now and was none when the set was composed may hold earlier rows among those left unread (no room key finds
    them); a task's lineage written since the marks were read leaves the pass's rooms on lineage the set was not composed
    under (``forgo``). A capture whose rows were read live (the stored facts stand for another source) left none unread.
    Either stands no more."""
    from ouroboros.store_hot_facts import lineage_writes

    if not (derived and derived.get(STANDS)):
        return False
    if lineage_writes(store.data_root) != derived[HELD]["marks"]["lineage"]:
        forgo(derived)
        return tail.complete
    if tail.complete and set(places) <= set(derived[HELD].get("places") or ()):
        return False
    derived[STANDS] = False
    return tail.complete


def standing(derived: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The marks a capture stands on, or ``None`` when it read the whole tail."""
    return derived[HELD] if derived and derived.get(STANDS) else None


def older_of(store: Any, held: Mapping[str, Any], pages: Iterable[tuple]) -> list:
    """The stored facts of the set's rows of each ``(chat, before, count)``: the ``count`` newest of that chat's rows
    before a position (``None``: its newest), the room key stored with each (``_compose``); a chat's in position order."""
    return _read(store, held, [(_OLDER, (chat, _END if before is None else before, count)) for chat, before, count in pages])


def copies_of(store: Any, held: Mapping[str, Any], words: Iterable[tuple]) -> list:
    """The stored facts of the set's rows with given words in given chats (``(chat, text_sha256)`` pairs, the digest
    stored with each row that has words), by position."""
    return _read(store, held, [(_COPIES, (json.dumps(sorted(set(words))),))])


def _read(store: Any, held: Mapping[str, Any], asks: Iterable[tuple]) -> list:
    """The stored facts of the set's rows each ``(selection, parameters)`` names, all in one transaction where the set
    and the stored tail are still the ones a capture stood on (``held``, its marks); ``Moved`` where they are not or
    cannot be read: that capture reads the whole tail again (``memory_inventory.over_one_state``)."""
    import sqlite3

    from ouroboros.chronicle_store import ChronicleStore

    rows: list = []
    try:
        with store._index() as db:
            if ChronicleStore._state(db, MARKS) != held or (ChronicleStore._state(db, "tail") or {}).get("source") != held.get("source"):
                raise Moved
            for positions, parameters in asks:
                sql, after = _FACTS.format(positions), -(1 << 63)
                while True:
                    text, last = db.execute(sql, (*parameters, after, _PAGE)).fetchone()
                    if last is None:
                        break
                    rows.extend(sorted(json.loads(text)))  # SQLite promises no aggregate order; the pages are in order
                    after = last
    except (TimeoutError, sqlite3.Error) as error:
        raise Moved from error
    return rows


def settle(store: Any, derived: Optional[Mapping[str, Any]], tail: Any, found: Any) -> None:
    """Count what this capture's pass read: on the set, the rows it left unread (``rows_cold``, less those a reader asks
    for later, ``memory_inventory.PlaceRows``) and the rows it read that no room shows open, each kept by a rule or new
    since the stored facts (``rows_kept``), or read only because the set, composed while they were still needed (open
    then), names them read until the next whole pass (``rows_stale``: a page sealed them since); with the open rows and
    the rows a reader asked for and was not given (``rows_ahead``), all it reads (law L7d). Else a pass over the whole
    tail (``tail_whole``). After a whole pass of a capture that read the marks over a complete stored tail, store what
    no reader used as the set, with those marks, in one index transaction. A busy index or a stored tail replaced
    meanwhile stores nothing: the next whole pass does."""
    from ouroboros.context_timing import counted

    snapshot = store._pinned()
    if derived and derived.get(STANDS):
        stored, writers = {row[0] for row in snapshot.tail_rows}, _writers(snapshot)
        shown = {entry[2] for entries in found.rooms.values() for entry in entries}
        counted("rows_cold", int(derived[HELD].get("rows") or 0))
        kept = [pos not in stored or pos in found.kept or _kept(meta, terminal, writers)
                for pos, _address, meta, terminal in tail.rows if pos not in shown]
        counted("rows_kept", sum(kept))
        if not all(kept):
            counted("rows_stale", len(kept) - sum(kept))
        return
    counted("tail_whole")
    if derived and derived.get(GATE) and tail.complete and snapshot is not None and not snapshot.proposed:
        _compose(store, snapshot, derived[GATE], tail, found)


def _writers(snapshot: Any) -> set:
    """Every task that wrote a page record, drafts and candidates too: a page links its writer's later terminal rows."""
    from ouroboros.memory_sealing import _terminal_writer

    return {_terminal_writer(snapshot.body(row)) for row in snapshot.records if row[2] == "page"} - {""}


def _kept(meta: Mapping[str, Any], terminal: Any, writers: set) -> bool:
    """A row the rules keep read whatever its rooms say: a quiet finish, an owner's Reply, a page writer's terminal row."""
    return (meta.get("quiet_finish") is True or meta.get("direction") == "in" and isinstance(meta.get("reply_to"), dict)
            or terminal is not None and str(meta.get("task_id") or "") in writers)


def _compose(store: Any, snapshot: Any, gate_: Gate, tail: Any, found: Any) -> None:
    import sqlite3

    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.memory_inventory import words_of

    writers = _writers(snapshot)
    # A terminal row's task is kept with it: a page its task writes later links it (``stands``); every row its room key,
    # the chat the room rule reads (``memory_inventory.rooms_of_row``), by the same function: ``older_of`` reads by it;
    # a row with words their digest as ``memory_view_conversations._repeats`` compares them (``copies_of``). Only a
    # row the stored tail holds as this pass read it (its position and identity) is named: a reader's page of the set
    # is then whole, and a row not stored yet, or projected anew since, stays read until the next whole pass.
    cold = [[pos, (meta.get("task_id") or None) if terminal is not None else None, _row_chat_id(meta),
             words_of(meta) or None, address["row_sha256"]]
            for pos, address, meta, terminal in tail.rows if pos not in found.kept and not _kept(meta, terminal, writers)]
    source, projects = (snapshot.state("tail") or {}).get("source"), {str(chat) for chat in gate_.facts.project_chat_ids}
    marks = {"marks": gate_.marks, "source": source, "sequence": snapshot.records[-1][0] if snapshot.records else 0,
             "places": sorted(found.places & projects)}
    try:
        with store._publishing(0) as db:
            held = store._state(db, "tail")
            if not isinstance(held, dict) or held.get("source") != source:
                return
            db.execute("DELETE FROM tail_cold")
            marks["rows"] = db.execute(
                "INSERT INTO tail_cold SELECT c.pos, c.task, c.chat, c.digest FROM (SELECT json_extract(value,'$[0]') AS "
                "pos, json_extract(value,'$[1]') AS task, json_extract(value,'$[2]') AS chat, json_extract(value,'$[3]') "
                "AS digest, json_extract(value,'$[4]') AS sha FROM json_each(?)) c JOIN tail_rows t ON t.pos=c.pos "
                "AND t.row_sha256=c.sha", (json.dumps(cold),)).rowcount
            store._set_state(db, MARKS, marks)
            db.commit()
    except (TimeoutError, sqlite3.Error):
        return
