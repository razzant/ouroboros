"""Disposable word postings for the chat stream inside ``state/dialog.sqlite``.

The postings only PROPOSE candidates: every candidate is decided by ``dialogue_words``' one predicate on the
original typed fields (``dialogue_recall``). A missing, partial, latched-unusable or unreadable generation, or a
SQLite without FTS5, leaves the reader on the scan with the same predicate and the same answer.

Objects (``_DDL``), all optional, created only by the maintenance child (``run``), never by a read, ``append``
or the import; no ``user_version`` change, so an older body opens, appends, edits and exports the file as before:

- ``dw_post``: contentless, positionless FTS5 (``content=''``, ``detail=none``, ``columnsize=0``, ``ascii``
  tokenizer over tokens already normalized by ``dialogue_words.tokens``); rowid = the chat row's ``seq``.
  A token is stored as at most its first ``TOKEN_CHARS`` code points: an encoding of very long words, never
  a limit on them (a query term is cut the same way and the original decides).
- ``dw_mark(segment, next_ord)``: per storage segment (an ``import_ledger`` unit), the ordinal below which its
  rows are posted. An archive that lands later below a completed live range is a new segment without a mark,
  so it is uncovered, never lost behind one global high-water mark.
- ``dw_dirty(seq, rev)``: rows whose text an edit changed after their posting, recorded by the trigger
  ``dw_edit`` inside the editing transaction; always re-read from the original, never deleted until a rebuild.
- ``dw_meta``: the generation (``<format>:<Unicode data version>:<uuid>``: the token rule's mark classes come
  from the running Python's Unicode data, so a body with other data bypasses the postings), its ``seen_rev`` witness of ``meta.chat_text_revision`` and ``usable``. The only
  writer that changes chat text on an existing row (``dialogue_store.revise_text``) updates the body, then
  increments the counter: the trigger requires its witness to equal the counter still unincremented, records
  the row at the next revision and advances the witness. Any other transition (an edit with no counter step, a
  counter that moved without a witnessed edit, a counter reset) latches ``usable=0`` or leaves ``seen_rev`` unequal
  to the counter: the generation is bypassed and the originals are scanned until a rebuild.
- ``dw_import``: one scalar step of ``import_epoch`` when a chat segment completes, so the parent's due check
  stays constant-cost.

The trigger uses ``json_extract``, which the store's own reads already require of this file (``dialogue_store``'s
``_PAGE``), and touches plain tables only: a connection without FTS5 still updates, appends and exports.
"""
from __future__ import annotations

import json
import logging
import pathlib
import sqlite3
import time
import unicodedata
import uuid
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

FORMAT = 1  # the token rule and encoding; a different one is bypassed and rebuilt
TOKEN_CHARS = 64
CHUNK_ROWS = 256  # rows one posting commit carries: a lock-hold bound measured in the report, not a policy
_PAGE_SQL_LIMIT = 500  # seqs one statement names
_DDL = (
    "CREATE TABLE IF NOT EXISTS dw_meta (id INTEGER PRIMARY KEY CHECK (id=1), format INTEGER NOT NULL, "
    "generation TEXT NOT NULL, seen_rev INTEGER NOT NULL, usable INTEGER NOT NULL CHECK (usable IN (0,1)), "
    "import_epoch INTEGER NOT NULL DEFAULT 0, built_epoch INTEGER NOT NULL DEFAULT -1)",
    "CREATE TABLE IF NOT EXISTS dw_mark (segment INTEGER PRIMARY KEY, next_ord INTEGER NOT NULL CHECK (next_ord>=0))",
    "CREATE TABLE IF NOT EXISTS dw_dirty (seq INTEGER PRIMARY KEY, rev INTEGER NOT NULL)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS dw_post USING fts5(tokens, content='', detail=none, columnsize=0, "
    "tokenize='ascii')",
    "CREATE TRIGGER IF NOT EXISTS dw_edit AFTER UPDATE OF body ON dialogue "
    "WHEN NEW.stream='chat' AND OLD.body IS NOT NEW.body AND "
    "json_extract(OLD.body,'$.text') IS NOT json_extract(NEW.body,'$.text') "
    "BEGIN "
    "UPDATE dw_meta SET usable = usable AND seen_rev = coalesce((SELECT CAST(value AS INTEGER) FROM meta "
    "WHERE key='chat_text_revision'),0), seen_rev = 1 + coalesce((SELECT CAST(value AS INTEGER) FROM meta "
    "WHERE key='chat_text_revision'),0) WHERE id=1; "
    "INSERT OR REPLACE INTO dw_dirty (seq, rev) SELECT NEW.seq, seen_rev FROM dw_meta WHERE id=1; "
    "END",
    "CREATE TRIGGER IF NOT EXISTS dw_import AFTER UPDATE OF state ON import_ledger "
    "WHEN NEW.stream='chat' AND NEW.state='done' AND OLD.state IS NOT NEW.state "
    "BEGIN UPDATE dw_meta SET import_epoch = import_epoch + 1 WHERE id=1; END",
)
_DROP = ("DROP TRIGGER IF EXISTS dw_edit", "DROP TRIGGER IF EXISTS dw_import", "DROP TABLE IF EXISTS dw_post",
         "DROP TABLE IF EXISTS dw_mark", "DROP TABLE IF EXISTS dw_dirty", "DROP TABLE IF EXISTS dw_meta")
_FTS5: Optional[bool] = None


def fts5_available() -> bool:
    """Whether this runtime's SQLite has FTS5 (asked once per process, on a private in-memory database)."""
    global _FTS5
    if _FTS5 is None:
        try:
            with sqlite3.connect(":memory:") as probe:
                probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            _FTS5 = True
        except sqlite3.Error:
            _FTS5 = False
    return _FTS5


def _revision(conn: sqlite3.Connection) -> int:
    record = conn.execute("SELECT value FROM meta WHERE key='chat_text_revision'").fetchone()
    return int(json.loads(record[0])) if record is not None else 0


def generation(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """The installed generation's facts with ``sound`` (usable now), or ``None`` when nothing is installed."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dw_meta'").fetchone() is None:
        return None
    record = conn.execute("SELECT format, generation, seen_rev, usable, import_epoch, built_epoch FROM dw_meta "
                          "WHERE id=1").fetchone()
    if record is None:
        return {"sound": False, "reason": "no generation row"}
    facts = dict(zip(("format", "generation", "seen_rev", "usable", "import_epoch", "built_epoch"), tuple(record)))
    revision = _revision(conn)
    triggers = {name for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('dw_edit','dw_import')")}
    unicode = str(facts["generation"]).split(":")[1] if str(facts["generation"]).count(":") >= 2 else ""
    reason = ("format" if facts["format"] != FORMAT else "latched unusable" if not facts["usable"]
              else f"unicode data {unicode or 'unrecorded'} != {unicodedata.unidata_version}"
              if unicode != unicodedata.unidata_version
              else f"witness {facts['seen_rev']} != chat_text_revision {revision}" if facts["seen_rev"] != revision
              else "trigger missing" if triggers != {"dw_edit", "dw_import"} else "")
    return {**facts, "revision": revision, "sound": not reason, "reason": reason}


def posting_text(row: Dict[str, Any]) -> str:
    """The tokens one row posts: every searchable field's tokens, cut to ``TOKEN_CHARS``, once each."""
    from ouroboros.dialogue_words import fields_of, tokens

    found = {token[:TOKEN_CHARS] for field in fields_of(row) if field.words for token in tokens(field.text)}
    return " ".join(sorted(found))


def match_expression(query: Any) -> str:
    """The FTS5 expression proposing a superset of the rows ``query`` can match: each alternative the AND of its
    prefixes (``"p"*``) and of every exact token of its phrases; adjacency and fields are the predicate's."""
    def term(token: str, prefix: bool) -> str:
        return '"' + token[:TOKEN_CHARS].replace('"', '""') + '"' + ("*" if prefix or len(token) > TOKEN_CHARS else "")

    alternatives = []
    for alternative in query.alternatives:
        parts = [term(prefix, True) for prefix in alternative.prefixes]
        parts += [term(token, False) for phrase in alternative.phrases for token in dict.fromkeys(phrase)]
        alternatives.append("(" + " AND ".join(parts) + ")")
    return " OR ".join(alternatives)


class Candidates(NamedTuple):
    """Coordinates the reader must re-check: posting hits, dirty rows and every uncovered row below ``upper``."""
    seqs: List[int]
    posted: int
    dirty: int
    uncovered: int
    generation: str
    plan: str  # what the snapshot binds: the generation, its marks and its dirty set


# The rows no posting covers yet: each chat unit from the ledger, then its rows at or above its mark through the
# (segment, ordinal) index. Never a walk of the chat history: with every row posted it reads one index probe a unit.
_UNCOVERED = ("SELECT d.seq FROM import_ledger l CROSS JOIN dialogue d INDEXED BY dialogue_ordinal "
              "WHERE l.stream='chat' AND d.segment=l.id "
              "AND d.ordinal >= coalesce((SELECT next_ord FROM dw_mark WHERE segment=l.id), 0) "
              "AND d.stream='chat' AND d.seq < ?")


def candidates(conn: sqlite3.Connection, query: Any, *, upper: int) -> Optional[Candidates]:
    """The candidate coordinates on this read connection, or ``None`` when the reader must scan (no FTS5, no
    sound generation, an expression the table refuses)."""
    if not fts5_available():
        return None
    try:
        facts = generation(conn)
        if not facts or not facts["sound"]:
            return None
        posted = {seq for (seq,) in conn.execute("SELECT rowid FROM dw_post WHERE dw_post MATCH ? AND rowid < ?",
                                                 (match_expression(query), int(upper)))}
        dirty = {seq for (seq,) in conn.execute("SELECT seq FROM dw_dirty WHERE seq < ?", (int(upper),))}
        uncovered = {seq for (seq,) in conn.execute(_UNCOVERED, (int(upper),))}
        marks = conn.execute("SELECT coalesce(group_concat(segment || ':' || next_ord, ','), '') FROM "
                             "(SELECT * FROM dw_mark ORDER BY segment)").fetchone()[0]
    except sqlite3.Error:
        log.debug("word postings unreadable; the reader scans", exc_info=True)
        return None
    return Candidates(sorted(posted | dirty | uncovered), len(posted), len(dirty), len(uncovered),
                      str(facts["generation"]), f"{facts['generation']}|{marks}|{len(dirty)}")


# ---- the maintenance child ------------------------------------------------------------------------

def _segments(conn: sqlite3.Connection) -> List[Tuple[int, int, int]]:
    """``(segment, mark, upper)`` of every chat unit holding rows: an archive's highest ordinal + 1 (its rows
    landed at once), the live unit's next ordinal now."""
    live = conn.execute("SELECT live, next_ordinal FROM streams WHERE stream='chat'").fetchone()
    units = []
    for (segment,) in conn.execute("SELECT id FROM import_ledger WHERE stream='chat' AND state IN ('done','live')"):
        if live is not None and segment == live[0]:
            upper = int(live[1])
        else:
            upper = int(conn.execute("SELECT coalesce(max(ordinal)+1, 0) FROM dialogue INDEXED BY dialogue_ordinal WHERE segment=?",
                                     (segment,)).fetchone()[0])
        mark = conn.execute("SELECT next_ord FROM dw_mark WHERE segment=?", (segment,)).fetchone()
        units.append((int(segment), int(mark[0]) if mark else 0, upper))
    return units


def due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Read-only and cheap: no install, archive walk or tokenization. Owed while FTS5 exists and nothing is
    installed, the generation is not sound, a completed segment is unbuilt or the live unit has unposted rows."""
    from ouroboros import dialogue_store as ds

    if row.get("state") == "failed":
        return child
    if not fts5_available() or not ds.present(root):
        return False
    try:
        with ds.transaction(root, write=False, wait_sec=0) as conn:
            facts_now = generation(conn)
            if facts_now is None or not facts_now["sound"]:
                return True
            if facts_now["import_epoch"] != facts_now["built_epoch"]:
                return True
            live = conn.execute("SELECT live, next_ordinal FROM streams WHERE stream='chat'").fetchone()
            if live is None:
                return False
            mark = conn.execute("SELECT next_ord FROM dw_mark WHERE segment=?", (live[0],)).fetchone()
            return (int(mark[0]) if mark else 0) < int(live[1])
    except Exception:
        log.debug("word postings facts unreadable; the next look asks again", exc_info=True)
        return False


def _install(root: pathlib.Path) -> str:
    """Create (or, for an unsound generation, recreate) the optional objects in one zero-wait write."""
    from ouroboros import dialogue_store as ds

    with ds.transaction(root, write=True, wait_sec=0) as conn:
        facts = generation(conn)
        if facts is not None and facts["sound"]:
            return str(facts["generation"])
        for statement in _DROP:
            conn.execute(statement)
        for statement in _DDL:
            conn.execute(statement)
        name = f"{FORMAT}:{unicodedata.unidata_version}:{uuid.uuid4().hex}"  # tokens depend on the mark classes
        conn.execute("INSERT INTO dw_meta (id, format, generation, seen_rev, usable) VALUES (1, ?, ?, ?, 1)",
                     (FORMAT, name, _revision(conn)))
        return name


def _wal_bytes(root: pathlib.Path) -> int:
    from ouroboros import dialogue_store as ds

    try:
        return ds.path(root).with_name(ds.path(root).name + "-wal").stat().st_size
    except OSError:
        return 0


def _log_chunk(root: pathlib.Path, **facts: Any) -> None:
    """One record a chunk in ``logs/supervisor.jsonl``, its own type beside the child's lifecycle records."""
    from ouroboros.utils import append_jsonl, utc_now_iso

    append_jsonl(pathlib.Path(root) / "logs" / "supervisor.jsonl",
                 {"ts": utc_now_iso(), "type": "maintenance_chunk", "job": "dialogue_words", **facts})


def _post_segment(root: pathlib.Path, name: str, segment: int, mark: int, upper: int,
                  should_stop: Callable[[], bool], totals: Dict[str, int]) -> bool:
    """Post one segment from its mark in chunks; False when stopped or contested (earlier chunks stay)."""
    from ouroboros import dialogue_store as ds

    while mark < upper:
        with ds.transaction(root, write=False) as conn:  # copied, then closed before any decoding
            page = conn.execute("SELECT json_group_array(json_array(seq, ordinal, body)) FROM (SELECT seq, ordinal, "
                                "body FROM dialogue INDEXED BY dialogue_ordinal WHERE segment=? AND stream='chat' AND ordinal>=? "
                                "AND ordinal<? "
                                "ORDER BY ordinal LIMIT ?)", (segment, mark, upper, CHUNK_ROWS)).fetchone()[0]
        records = json.loads(page or "[]")
        reach = records[-1][1] + 1 if len(records) == CHUNK_ROWS else upper
        postings, source_bytes = [], 0
        for seq, _ordinal, body in records:
            source_bytes += len(body.encode("utf-8"))
            text = posting_text(ds.decode(body))
            if text:
                postings.append((int(seq), text))
        started = time.monotonic()
        try:
            with ds.transaction(root, write=True, wait_sec=0) as conn:
                acquired = time.monotonic()
                waited = acquired - started
                facts = generation(conn)
                held = conn.execute("SELECT next_ord FROM dw_mark WHERE segment=?", (segment,)).fetchone()
                if not facts or facts["generation"] != name or (int(held[0]) if held else 0) != mark:
                    return False  # another worker, or a rebuild: its work stands, this chunk is not ours
                conn.executemany("INSERT INTO dw_post (rowid, tokens) VALUES (?, ?)", postings)
                conn.execute("INSERT OR REPLACE INTO dw_mark (segment, next_ord) VALUES (?, ?)", (segment, reach))
                dirty = conn.execute("SELECT count(*) FROM dw_dirty").fetchone()[0]
            hold = time.monotonic() - acquired  # from the write lock to its commit; tokenizing came before
        except ds.StoreBusy:
            _log_chunk(root, generation=name, segment=segment, from_ord=mark, next_ord=mark, rows_read=len(records),
                       outcome="contended", wait_ms=round((time.monotonic() - started) * 1000, 1))
            raise
        token_bytes = sum(len(text.encode("utf-8")) for _seq, text in postings)
        totals["rows"] += len(records)
        totals["source_bytes"] += source_bytes
        totals["token_bytes"] += token_bytes
        totals["slowest_ms"] = max(totals["slowest_ms"], round(hold * 1000, 1))
        _log_chunk(root, generation=name, segment=segment, from_ord=mark, next_ord=reach, rows_read=len(records),
                   rows_indexed=len(postings), source_bytes=source_bytes, token_bytes=token_bytes, dirty_rows=dirty,
                   wait_ms=round(waited * 1000, 1), hold_ms=round(hold * 1000, 1), wal_bytes=_wal_bytes(root),
                   outcome="committed")
        mark = reach
        if should_stop():
            return False
    return True


def run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """Install when owed, then post every chat segment from its mark; ``done`` once every completed segment
    and the live unit are posted at one import epoch. A turn never runs this: the maintenance child does."""
    from ouroboros import dialogue_store as ds

    root = pathlib.Path(root)
    if not fts5_available() or not ds.present(root):
        return {"state": "done", "items": 0}
    name = _install(root)
    totals = {"rows": 0, "source_bytes": 0, "token_bytes": 0, "slowest_ms": 0.0}
    with ds.transaction(root, write=False) as conn:
        epoch = conn.execute("SELECT import_epoch FROM dw_meta WHERE id=1").fetchone()[0]
        units = _segments(conn)
    for segment, mark, upper in units:
        if not _post_segment(root, name, segment, mark, upper, should_stop, totals):
            return {"state": "pending", "items": totals["rows"], **totals}
    with ds.transaction(root, write=True, wait_sec=0) as conn:
        conn.execute("UPDATE dw_meta SET built_epoch=? WHERE id=1 AND import_epoch=? AND generation=?",
                     (epoch, epoch, name))
    state = "done" if not due(root, {}, {}, True) else "pending"
    return {"state": state, "items": totals["rows"], **totals}


# ---- retirement -----------------------------------------------------------------------------------

def retire(root: Any) -> Dict[str, Any]:
    """Drop every optional object in one write; the canonical rows are untouched. Returns the pages freed
    for reuse inside ``state/dialog.sqlite`` (the file does not shrink: no vacuum). Without FTS5 the virtual
    table cannot be dropped: the generation is latched unusable (the reader scans) and the objects stay
    coherent until a capable runtime retires them (``state="fts5_required"``)."""
    from ouroboros import dialogue_store as ds

    root = pathlib.Path(root)
    if not ds.present(root):
        return {"state": "absent", "pages_freed": 0, "bytes_freed": 0}
    with ds.transaction(root, write=True) as conn:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        before = conn.execute("PRAGMA freelist_count").fetchone()[0]
        present = conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'dw\\_%' ESCAPE '\\'").fetchone()[0]
        if not fts5_available() and conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='dw_post'").fetchone() is not None:
            conn.execute("UPDATE dw_meta SET usable=0 WHERE id=1")
            return {"state": "fts5_required", "objects": present, "pages_freed": 0, "bytes_freed": 0}
        for statement in _DROP:
            conn.execute(statement)
        after = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return {"state": "retired" if present else "nothing_installed", "objects": present, "page_size": page_size,
            "pages_freed": after - before, "bytes_freed": (after - before) * page_size,
            "file_shrinks": False}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m ouroboros.dialogue_word_index --retire --data-root <root>``: the explicit rollback."""
    import argparse

    parser = argparse.ArgumentParser(prog="dialogue_word_index")
    parser.add_argument("--retire", action="store_true", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(retire(pathlib.Path(args.data_root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
