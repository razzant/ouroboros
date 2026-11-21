"""When each room last had a message: the time the sidebar orders Projects by (DESIGN Shell "Navigation").

``counts`` reads a stored chat row's declared fields: the owner's words and answers, Ouroboros's standalone
reply, question, delivery or reminder, and a root's final or terminal incident count; any other System row (a
closure, a pause, a receipt), a task card's line, a child's words (also recovered from task lineage), and a task's summary row
(a technical projection, also of a salvaged end) do not. ``state/dialog.sqlite`` keeps, per room, the newest
original event time (``ts``) of such a row the room shows (``room_membership``): an append dates the rooms it is
shown in, an archive import its segment's, the appearance job the rooms it derives for a row later, a binding (or
the origin a ref-less one gains) only its admitted delta, and the maintenance job dates the history once per
``GENERATION`` and, whenever the bindings file is not the one the store last covered, every room bound since its last
pass or naming an origin newer than its time (``BINDINGS_KEY``: a save that dated its room moves that witness only
from exactly its own start). ``THROUGH_KEY`` covers appends; old-body writes leave the job owed. Only a new generation
replaces incorrect old maxima. The public revision hashes Project times; technical activity, unread revisions,
raw rows and read cursors stay what they were. Retained binding origins share history's canonical reader.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ouroboros.subagent_messages import DELIVERY_KINDS, is_conversation_card_content, is_task_card_message

log = logging.getLogger(__name__)

JOB = "room_messages"
GENERATION = 2  # recovered child lineage: replace maxima an older rule incorrectly raised
GENERATION_KEY, REVISION_KEY = "room_messages_generation", "room_messages_revision"  # the store's meta
BINDINGS_KEY = "room_messages_bindings"  # meta: identity of the bindings file every binding in it dated its room under
THROUGH_KEY = "room_messages_through"  # chat coordinate below which this rule dated every row
DDL = "CREATE TABLE IF NOT EXISTS room_messages (chat_id INTEGER PRIMARY KEY, at TEXT NOT NULL)"
_HOST_ROWS = frozenset({"relay_entry", "routing_options", "task_summary"})  # beside the conversation, not in it
_SYSTEM_MESSAGES = frozenset({"terminal_incident", "reminder"})  # System rows written for the owner to read
_FINAL = re.compile(r"final:[^:]+:[^:]+")  # a task's own final delivery (``terminal_delivery.delivery_id_for``)
_PAGE = 32


def counts(row: Any, child=lambda _row: False) -> bool:
    """Whether a stored chat row is a message of its conversation (module docstring)."""
    kind = str(row.get("type") or "")
    if row.get("direction") == "in" or kind == "quiz_answer":
        return True
    if kind in _HOST_ROWS or is_conversation_card_content(row, kind, child):
        return False
    if row.get("direction") == "system" or row.get("role") == "system":
        return kind in _SYSTEM_MESSAGES or bool(_FINAL.fullmatch(str(row.get("delivery_id") or "")))
    return row.get("direction") == "out" and (str(row.get("text") or "").strip() not in ("", "​")
                                              or kind in DELIVERY_KINDS)


def classified(rows: Iterable[Any], rooms: Any) -> List[Any]:
    """Count with history's recovered lineage, using only indexed task facts, before any writer lock.
    An unavailable lineage read leaves the caller's coverage owed; it never certifies a guessed root.
    A speaker without a typed role may be a child an older body wrote with its role only in ``metadata``:
    dating alone reads that scalar (``_legacy_roles``); the room rule's lineage stays body-free."""
    from ouroboros.store import StoreError

    candidates = [row for row in rows if counts(row)]
    wanted = set()

    def collect(row):
        if row.get("task_id") and not row.get("delegation_role"):
            wanted.add(str(row["task_id"]))
        return False

    for row in candidates:
        counts(row, collect)  # asks only for the speaker of possible card content, never an owner or delivery
    if wanted and not rooms.warm(wanted):
        raise StoreError("message lineage unavailable")
    untyped = {tid for tid in wanted if not rooms.lineage(tid).get("delegation_role")}
    legacy = _legacy_roles(rooms, untyped) if untyped else {}
    if legacy is None:
        raise StoreError("message lineage unavailable")
    return [row for row in candidates if counts(row, lambda entry: (
        str(entry.get("task_id") or "") in wanted
        and is_task_card_message({**rooms.lineage(entry["task_id"]), **legacy.get(str(entry["task_id"]), {})})))]


# A role an older body kept only in the task's ``metadata``, for tasks without a typed one: one statement, and
# only the scalar leaves SQLite, never the task document.
_LEGACY_ROLE = ("SELECT task_id, json_extract(body, '$.metadata.delegation_role') AS delegation_role "
                "FROM task_bodies WHERE task_id IN (SELECT value FROM json_each(?))")


def _legacy_roles(rooms: Any, task_ids: Iterable[str]) -> Optional[Dict[str, Dict[str, str]]]:
    """``{task id: {"delegation_role": role}}`` for those that keep one in ``metadata``; ``None`` when the store
    cannot answer now (a body that is not JSON among them, too), except that a strict reader keeps contention typed,
    as the lineage read does (``projects_registry._read_lineage``)."""
    from ouroboros.store import StoreBusy, open_store
    from ouroboros.store_capture import capture

    try:
        with open_store(rooms.root, write=False) as txn:
            captured = capture(txn.conn, _LEGACY_ROLE, (json.dumps(sorted(task_ids)),),
                               ("task_id", "delegation_role"))
        records = captured.tuples()
    except StoreBusy as exc:
        if rooms.strict and exc.reason == "contention":
            raise
        log.debug("legacy task role read failed", exc_info=True)
        return None
    except Exception:
        log.debug("legacy task role read failed", exc_info=True)
        return None
    return {str(tid): {"delegation_role": str(role)} for tid, role in records if role}


def note(conn: Any, times: Iterable[Tuple[Any, str]]) -> None:
    """Raise rooms' times inside the caller's transaction; the revision grows when one moved."""
    pairs = sorted({(int(chat), str(at)) for chat, at in times if chat is not None and int(chat) > 0 and at})
    if not pairs:
        return
    conn.execute(DDL)
    before = conn.total_changes
    conn.executemany("INSERT INTO room_messages (chat_id, at) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE "
                     "SET at=excluded.at WHERE excluded.at > room_messages.at", pairs)
    if conn.total_changes != before:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, '1') ON CONFLICT(key) DO UPDATE "
                     "SET value=CAST(value AS INTEGER)+1", (REVISION_KEY,))


def shown(entries: Iterable[Any], projects: Any, rooms: Any) -> List[Tuple[int, str]]:
    """``(room, time)`` of each message among ``entries`` (``ds.Row``) for every Project the room rule shows it in:
    what ``room_appearances`` owes the rooms it derives for a row after its write (or an older body's rows)."""
    from ouroboros.room_membership import row_rooms

    pairs: List[Tuple[int, str]] = []
    for row in classified((entry.row for entry in entries), rooms):
        try:
            written = int(row.get("chat_id", 1))
        except (TypeError, ValueError):
            written = 1
        pairs.extend((chat, str(row.get("ts") or "")) for chat in row_rooms(written, row, projects, rooms)
                     if chat in projects)
    return pairs


def read(root: Any) -> Tuple[Optional[Dict[int, str]], Optional[str]]:
    """``(times by chat id, revision)``, both ``None`` while this generation has not dated the history: the order
    is then pending, never guessed. The revision moves whenever a time does and when the history is dated."""
    from ouroboros import dialogue_store as ds

    with ds.transaction(root, write=False) as conn:
        if ds._meta(conn, GENERATION_KEY) != GENERATION:
            return None, None
        revision = f"{GENERATION}.{int(ds._meta(conn, REVISION_KEY) or 0)}"
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='room_messages'").fetchone():
            return {}, revision
        return {int(row["chat_id"]): str(row["at"]) for row in conn.execute("SELECT chat_id, at FROM room_messages")}, revision


def project_revision(root: Any, projects: Iterable[Any]) -> Optional[str]:
    """The public refresh token: only the list's Project times, never Main/place-only writes.
    The store's own revision still invalidates publication after maintenance corrects a time."""
    times, _revision = read(root)
    if times is None:
        return None
    values = sorted((int(project["chat_id"]), times.get(int(project["chat_id"])) or str(project.get("created_at") or ""))
                    for project in projects)
    return f"{GENERATION}." + hashlib.sha256(json.dumps(values).encode()).hexdigest()


def _times(conn: Any, keys: Iterable[Tuple[str, Any]], upper: int) -> Dict[int, str]:
    """``{seq: ts}`` of every chat row the probe keys address below ``upper``, from each key's index
    (``dialogue_store._PROBE_INDEX``) and the row's own ``ts`` column: no body is read."""
    from ouroboros import dialogue_store as ds

    seqs = ds._probe_seqs(conn, "chat", keys, before=upper)
    return {int(seq): str(ts or "") for seq, ts in conn.execute(
        "SELECT seq, ts FROM dialogue WHERE stream='chat' AND seq IN (SELECT value FROM json_each(?))",
        (json.dumps(sorted(seqs)),))} if seqs else {}


def newest(root: Any, chat: int, *, binding: Optional[dict] = None, upper: Optional[int] = None) -> str:
    """The newest original time of a message a Project shows: every candidate of its canonical read
    (``chat_chain.room_keys``: its own rows, late bindings, rows shown from elsewhere, the owner rows its bindings
    name) by ``(seq, ts)``, then bodies in that time's order to the first one that counts and is the room's. The
    order rows were written in never stands in for their time; bodies past the answer stay unread."""
    from ouroboros import chat_chain, dialogue_store as ds
    from ouroboros.project_dialogue import project_origin_rows
    from ouroboros.projects_registry import RoomResolver

    keys, identities, admits, upper = chat_chain.room_keys(root, chat, binding=binding, upper=upper)
    retained = max((str(row["ref"].get("ts") or "") for row in project_origin_rows(
        root, chat, bindings=[binding] if binding is not None else None)), default="")
    rooms = RoomResolver(root, strict=True)
    with ds.transaction(root, write=False) as conn:
        times = _times(conn, keys, upper)
    times.update((seq, str(entry.row.get("ts") or "")) for seq, entry in identities.items())
    order = sorted(times, key=lambda seq: (times[seq], seq), reverse=True)
    for start in range(0, len(order), _PAGE):
        page = {**ds.rows_at(root, "chat", order[start:start + _PAGE]), **identities}
        counted = {id(row) for row in classified((entry.row for entry in page.values()), rooms)}
        for seq in order[start:start + _PAGE]:
            entry = page.get(seq)
            if times[seq] <= retained:
                return retained
            if entry is not None and id(entry.row) in counted and admits(entry.row):
                return times[seq]
    return retained


def reread(root: Any, chat: int, *, replace: bool = False) -> bool:
    """Date one Project; a new generation corrects old maxima, only under the stream end it read."""
    from ouroboros import dialogue_store as ds

    upper = ds.stream_end(root, "chat")
    at = newest(root, chat, upper=upper)
    with ds.transaction(root, write=True) as conn:
        if replace:
            if conn.execute("SELECT next_seq FROM streams WHERE stream='chat'").fetchone()[0] != upper:
                return False  # a concurrent append may be below the incorrect old maximum: read again
            conn.execute(DDL)
            conn.execute("DELETE FROM room_messages WHERE chat_id=? AND at!=?", (chat, at))
        if at:
            note(conn, [(chat, at)])
    return True


def bindings_identity(root: Any) -> Optional[list]:
    """The bindings file's identity (``utils.file_identity``), ``None`` while there is none."""
    from ouroboros import utils
    from ouroboros.projects_registry import _bindings_path

    try:
        return list(utils.file_identity(_bindings_path(root).stat()))
    except FileNotFoundError:
        return None


def bound(root: Any, binding: dict, start: Optional[list], end: Optional[list]) -> None:
    """A binding or its enrichment (the bindings file going from identity ``start`` to ``end``) makes its task tree's
    rows and the owner message it names its Project's: date the room now, without waiting for a lock, and move the
    witness on from exactly ``start``. A store that cannot answer at once leaves the witness behind: the server's next
    look finds the job owed. A root with no store yet has nothing stored to date. Never raises."""
    from ouroboros import dialogue_store as ds, store

    chat = int(binding["project_chat_id"])
    if not ds.present(root):
        return
    try:
        with store.acquisition_budget(0):
            at = newest(root, chat, binding=binding)
            with ds.transaction(root, write=True, wait_sec=0) as conn:
                note(conn, [(chat, at)])
                conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                             (json.dumps(end), BINDINGS_KEY, json.dumps(start)))
    except Exception:
        log.debug("Room %s waits for the room_messages job", chat, exc_info=True)


def _owed(root: Any, through: str, times: Dict[int, str]) -> Tuple[set, str]:
    """The Project rooms bindings owe a reread and the newest binding time (``project_task_bindings``): bound after
    ``through``, or naming an origin newer than the room's time in ``times`` (``source_ref.ts``, the owner row's own
    time: an enrichment adds it to the room and keeps the binding's time)."""
    from ouroboros.projects_registry import project_task_bindings

    rooms, newest_at = set(), through
    for binding in project_task_bindings(root).values():
        chat, at = int(binding.get("project_chat_id") or 0), str(binding.get("bound_at") or "")
        ref = binding.get("source_ref")
        if chat <= 0:
            continue
        if at > through:
            rooms.add(chat)
            newest_at = max(newest_at, at)
        elif isinstance(ref, dict) and str(ref.get("ts") or "") > times.get(chat, ""):
            rooms.add(chat)
    return rooms, newest_at


def due(root: Any, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """History this generation has not dated (once the archive is in), or a bindings file the store has not covered
    (one stat and one meta read: no bindings are read here)."""
    from ouroboros import dialogue_store as ds

    if row.get("state") == "failed":
        return child
    try:
        if not ds.present(root):
            return False
        try:
            ds.require_imported(root, "chat")
        except ds.StoreMigrating:
            return False  # dated once the archive arrived; a segment that cannot be imported holds nothing to date
        except ds.StoreError:
            pass
        through = ds.meta(root, THROUGH_KEY)
        return (ds.meta(root, GENERATION_KEY) != GENERATION or ds.meta(root, BINDINGS_KEY) != bindings_identity(root)
                or type(through) is not int or through < ds.stream_end(root, "chat"))
    except Exception:
        log.debug("Room message facts unreadable; the next look asks again", exc_info=True)
        return False


def run(root: Any, should_stop: Any) -> Dict[str, Any]:
    """Date every Project once per generation, checkpointing chat id and the pass's original input bounds.
    Resuming covers no newer debt behind that cursor; later passes date owed bindings and append gaps."""
    from ouroboros import dialogue_store as ds, maintenance_jobs
    from ouroboros.projects_registry import reserved_project_chat_ids

    row, items, witness = maintenance_jobs.job(root, JOB) or {}, 0, bindings_identity(root)
    end = ds.stream_end(root, "chat")
    rooms, through = _owed(root, str(row.get("bound_through") or ""), read(root)[0] or {})
    dating = ds.meta(root, GENERATION_KEY) != GENERATION
    cursor, history = 0, None
    if dating:
        history = row.get("history") if row.get("generation") == GENERATION else None
        if isinstance(history, dict):
            cursor = int(row.get("cursor") or 0)
        else:
            # A resumed prefix covers only the pass's original inputs. New bindings or old-body
            # appends behind its cursor remain owed after this generation becomes readable.
            history = {"end": end, "bindings": witness, "bound_through": through}
        maintenance_jobs.record(root, JOB, state="pending", cursor=cursor, generation=GENERATION, history=history)
        rooms = {chat for chat in reserved_project_chat_ids(root, strict=True) if chat > cursor}
    for chat in sorted(rooms):
        # A concurrent append need not throw away the completed prefix or wait for the next sweep.
        # One immediate reread handles the ordinary collision; repeated contention yields with its cursor.
        if not reread(root, chat, replace=dating) and not (dating and reread(root, chat, replace=True)):
            if dating:
                maintenance_jobs.record(root, JOB, state="pending", cursor=cursor, generation=GENERATION, history=history)
            return {"state": "pending", "items": items}
        items += 1
        cursor = chat
        stop = should_stop()
        if dating and (stop or items % _PAGE == 0):
            maintenance_jobs.record(root, JOB, state="pending", cursor=cursor, generation=GENERATION, history=history)
        if stop:
            return {"state": "pending", "items": items}
    if dating:
        end, witness, through = history["end"], history["bindings"], history["bound_through"]
    elif not _catch_up(root, end, should_stop):
        return {"state": "pending", "items": items}
    with ds.transaction(root, write=True) as conn:
        if dating:
            ds._set_meta(conn, GENERATION_KEY, GENERATION)
            ds._set_meta(conn, THROUGH_KEY, end)
        ds._set_meta(conn, BINDINGS_KEY, witness)
    maintenance_jobs.record(root, JOB, state="done", cursor=0, bound_through=through, history=None)
    return {"state": "done", "items": items}


def _catch_up(root: Any, end: int, should_stop: Any) -> bool:
    """Rows an older body appended after this one, using the append coverage pattern; a busy
    append's unresolved lineage or membership leaves the same debt. No binding/history rescan."""
    from ouroboros import dialogue_store as ds
    from ouroboros.room_appearances import _rule

    projects, rooms = _rule(root)
    with ds.transaction(root, write=True) as conn:
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')", (THROUGH_KEY,))
        start = ds._meta(conn, THROUGH_KEY)
    while start < end:
        with ds.transaction(root, write=False) as conn:
            page = ds._page(conn, "WHERE stream='chat' AND seq>=? AND seq<? ORDER BY seq LIMIT 500", (start, end))
        entries = [ds._row(record) for record in ds._decoded(page)]
        upto = entries[-1].seq + entries[-1].length if len(entries) == 500 else end
        dated = shown(entries, projects, rooms)
        with ds.transaction(root, write=True) as conn:
            note(conn, dated)
            if not conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                                (json.dumps(upto), THROUGH_KEY, json.dumps(start))).rowcount:
                return False
        start = upto
        if should_stop() and start < end:
            return False
    return True
