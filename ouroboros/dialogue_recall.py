"""``chat_history(words=…)``: words over the conversation, read only when the mind calls.

One read captures a frozen upper bound of the chat stream and the legacy reader's coverage (import and
invalid-interval gaps, a room's appearance gaps, the frozen consolidation cursor, the durable gap blocks:
``capture_coverage``). Its candidates are the rows of that capture the exact filters keep
(``memory.chat_history_filter``; A2A traffic never); each is re-checked by the one predicate
(``dialogue_words``) on its typed fields, an alternative inside one field.

Two paths give the same answer. A read of every room takes its candidates from the word postings
(``dialogue_word_index``) when they are sound: posting hits, edited rows and every row not yet posted, in
one read transaction; ``RECHECK_BOUND`` candidates at most are re-checked a call, taken from the asked end
of time, and a bounded answer names the examined range and a continuation (``snapshot=<token>:<n>``) that
always advances. Otherwise (a room read, no FTS5, postings not built or bypassed) the scan re-checks every
row it reads, as the legacy substring search reads every row; no bound stands there, so a default call
returns the newest matches whatever the store's size. The header names the path that answered.

Order. Hits are ordered by recorded time (``ts`` parsed as ISO-8601), ties by stored sequence; a row
whose time cannot be read stands at the oldest end, by stored sequence. ``order`` is the caller's direction:
newest (default) or oldest first. No relevance order and no host choice of a row.

Each hit is resolved for its page only: the rooms that show it now (``room_membership.row_rooms`` over the
project registry, the rule every room reader uses), its acting seals there, its author as
``memory_read(rows=true)`` names it (the lineage epoch and the row's stream position), and a reader that
returns it (one of its showing rooms; a row no room shows is read by its identity, ``source_ref``).
``voice=owner`` selects the rows ``dialogue_provenance.owner_speech`` calls my human's words; every typed
field of such a row is matched and each keeps its own speaker. A row stamped by the API door
(``client_surface.channel == "api_command"``) says that its sender is not recorded; any other row claims
nothing about its route.

The snapshot binds the capture's upper bound, ``meta.chat_text_revision`` (an in-place edit refuses a
shifted page), the durable gap blocks and the query. Residual, named: the room registry and the owner's
confirmed place accounts have no cheap revision, so a task rebinding between two pages can shift a page.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.dialogue_words import Field, WordQuery, excerpt, fields_of, match_fields, parse

log = logging.getLogger(__name__)

# Postings candidates one call re-checks at most: a bound on one read's time, never a choice of which rows
# matter. Measured on a generated 100,000-row store: about 35 µs a candidate, so 50,000 keep one call under
# about two seconds whatever the query; met, it is disclosed and the continuation advances.
RECHECK_BOUND = 50_000
EXCERPT_CHARS = 240  # the shown part of one field around its first match; the row's reader holds it whole
ORDERS = ("newest", "oldest")
VOICES = ("owner",)
API_DOOR_NOTE = "API door, sender not recorded"
OWNER_ACCOUNTS_UNREAD = "owner accounts unreadable: no place row is labelled as my human's"
SPEAKER_WORDS = {"owner": "my human", "host": "host"}  # the mind's labels for those field speakers
_CHUNK = 500  # SQL variables one statement names
# The owner-facing sentence of ``chat_history``'s ``words`` (tools/control.py), beside the reader it describes.
WORDS_DESCRIPTION = (
    "Each item is one alternative (any may match): word prefixes that must all occur, in any order, in one field "
    "of one row (its text, a card's question, an option, the comment); a quoted part is an exact word or phrase. "
    "search instead finds a literal piece of the rendered row (an id, a label, part of a word, text written "
    "without spaces).")


def _room(room_id: Any) -> Optional[int]:
    text = str(room_id if room_id is not None else "").strip()
    if not text:
        return None
    if not text.lstrip("-").isdigit():
        raise ValueError(f"room_id {text!r} is not a room address: a room is its chat id as text (1 is Main), "
                         "as memory_read and my memory view print it")
    return int(text)


def check_arguments(words: Any, *, search: str = "", room_id: Any = "", voice: str = "", order: str = "",
                    filters: Optional[Mapping[str, str]] = None) -> Tuple[Optional[WordQuery], Optional[int]]:
    """``(query or None, room or None)`` for one ``chat_history`` call; ``ValueError`` fit for the caller.

    ``words`` and the legacy ``search`` (any non-empty text, whitespace included) exclude each other;
    ``room_id``, ``voice`` and ``order`` go with ``words`` only, so a call without ``words`` keeps the legacy
    reader's contract exactly. With ``words`` the exact ``filters`` are checked here too (a date that is not
    ISO-8601), before anything is read."""
    if words is None or words == []:
        if words == []:
            parse(words)  # the empty query's own message
        given = [name for name, value in (("room_id", room_id), ("voice", voice), ("order", order))
                 if str(value if value is not None else "").strip()]
        if given:
            raise ValueError(f"{', '.join(given)} {'go' if len(given) > 1 else 'goes'} with words=[…]; "
                             "the substring search takes none of them")
        return None, None
    if str(search or ""):
        raise ValueError("words and search exclude each other: words=[…] matches word prefixes within one field, "
                         "search=\"…\" keeps the old substring filter; give one")
    if str(voice or "") not in ("", *VOICES):
        raise ValueError(f"voice is {' or '.join(VOICES)} (omit it for every speaker); got {voice!r}")
    if str(order or "") not in ("", *ORDERS):
        raise ValueError(f"order is newest or oldest (default newest); got {order!r}")
    query, room = parse(words), _room(room_id)
    from ouroboros.memory import chat_history_filter

    chat_history_filter(filters or {}, "")
    return query, room


def capture_coverage(memory: Any, *, upper: int, imported: Mapping[str, Any], matched: int,
                     shown: Any = ()) -> Dict[str, Any]:
    """The legacy reader's coverage of one capture below ``upper`` (``Memory.read_chat_generations`` and both
    paths of the words reader build it here): skipped intervals, import state, a room's appearance gaps, the
    frozen consolidation cursor and the durable gap blocks."""
    from ouroboros import chat_chain, dialogue_store as ds, dialogue_positions as dp

    gaps = [{"kind": "jsonl_malformed" if gap.kind == "invalid" else "jsonl_non_object",
             "path": str(ds.path(memory.drive_root)), "offset": gap.seq, "length": gap.length}
            for gap in dp.skipped_between(memory.drive_root, "chat", upper=upper)]
    if imported["pending"]:
        gaps.append({"kind": "history_importing", "detail": "Dialogue history is still importing"})
    if imported["mismatch"]:
        gaps.append({"kind": "history_import_failed", "detail": "A segment could not be imported"})
    if shown:
        from ouroboros.room_appearances import GAP_DETAILS

        gaps.extend({"kind": kind, "detail": GAP_DETAILS[kind]} for kind in sorted(shown))
    coverage = {
        "generations": ([{"path": str(ds.path(memory.drive_root)), "first_line_sha256": "dialogue:chat",
                          "size": upper, "rows": matched}] if upper else []),
        "matched_rows": matched, "gaps": gaps, "capture_attempts": 1,
        "snapshot_changed_during_read": False, "snapshot_stable": bool(imported["imported"]),
        "reader": "chat_history(count, offset, search)",
    }
    if imported["imported"]:
        _position, missing = chat_chain.generation_cursor(memory.drive_root, memory.load_dialogue_meta())
        if missing:
            gaps.append({"kind": "consolidation_cursor_generation_missing",
                         "detail": "The consolidation cursor names a generation that is no longer readable."})
    durable_gaps, durable_gap_ids = memory._durable_dialogue_gaps()
    gaps.extend(durable_gaps)
    coverage["durable_gap_ids"] = durable_gap_ids
    return coverage




def time_key(row: Mapping[str, Any], seq: int) -> Tuple[int, float, int]:
    """The hit order: recorded time, then stored sequence; an unreadable time at the oldest end."""
    from ouroboros.memory import _history_timestamp  # the legacy reader's own parse of ``ts``

    try:
        return 1, _history_timestamp(row.get("ts")).timestamp(), int(seq)
    except (ValueError, TypeError):
        return 0, 0.0, int(seq)


def _text_revision(root: pathlib.Path) -> int:
    from ouroboros import dialogue_store as ds

    return int(ds.meta(root, "chat_text_revision") or 0)


def _scan(memory: Any, room: Optional[int], keep: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    """Every row of the capture (one room's by the room rule, else every room's) that ``keep`` admits, with
    its stored sequence: the scan path, which re-checks every row it reads."""
    from ouroboros import chat_chain, dialogue_store as ds
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    root = pathlib.Path(memory.drive_root)
    upper = ds.stream_end(root, "chat")
    imported = ds.range_coverage(root, "chat", upper=upper)
    shown: set = set()
    source = (ds.iter_rows(root, "chat", upper=upper) if room is None else
              chat_chain.room_rows(root, room, upper=upper, gaps=shown))
    pairs = [(entry.seq, entry.row) for entry in source
             if not is_a2a_chat_id(entry.row.get("chat_id")) and keep(entry.row)]
    coverage = capture_coverage(memory, upper=upper, imported=imported, matched=len(pairs), shown=shown)
    return {"pairs": pairs, "coverage": coverage, "bounded": False, "plan": "",
            "path": "scan of every row" if room is None else f"scan of room {room}"}


def _posted(memory: Any, query: WordQuery, *, newest: bool, start: int) -> Optional[Dict[str, Any]]:
    """The postings path for a read of every room: in ONE read transaction the candidates below a frozen
    upper bound, ordered by ``time_key`` from the asked end, cut to ``[start, start + RECHECK_BOUND)`` and then to
    the end of that last recorded time, and the original rows of that window. ``None``: no sound postings here;
    the caller scans. ``{"malformed": True}``: ``start`` is no boundary a window of this plan ends at."""
    from ouroboros import dialogue_store as ds, dialogue_word_index as dwi

    root = pathlib.Path(memory.drive_root)
    upper = ds.stream_end(root, "chat")
    imported = ds.range_coverage(root, "chat", upper=upper)
    with ds.transaction(root, write=False) as conn:
        found = dwi.candidates(conn, query, upper=upper)
        if found is None:
            return None
        keyed: List[Tuple[Tuple[int, float, int], str]] = []
        for at in range(0, len(found.seqs), _CHUNK):
            chunk = found.seqs[at:at + _CHUNK]
            keyed += [(time_key({"ts": ts}, seq), str(ts or "")) for seq, ts in conn.execute(
                f"SELECT seq, ts FROM dialogue WHERE stream='chat' AND seq IN ({','.join('?' * len(chunk))})", chunk)]
        keyed.sort(reverse=newest)
        if start and (start >= len(keyed) or keyed[start - 1][0][:2] == keyed[start][0][:2]):
            return {"malformed": True}  # no window this plan issued begins there
        end = min(len(keyed), start + RECHECK_BOUND)
        while end < len(keyed) and keyed[end][0][:2] == keyed[end - 1][0][:2]:
            end += 1  # a window never splits rows of one recorded time: copies of one row stay together
        window = keyed[start:end]
        wanted = sorted(key[2] for key, _ts in window)
        pages = [ds._page(conn, f"WHERE stream='chat' AND seq IN ({','.join('?' * len(chunk))})", tuple(chunk))
                 for chunk in (wanted[at:at + _CHUNK] for at in range(0, len(wanted), _CHUNK))]
    pairs = [(entry.seq, entry.row) for entry in (ds._row(record) for record in ds._decoded(*pages))]
    coverage = capture_coverage(memory, upper=upper, imported=imported, matched=len(pairs))
    span = (f"{len(window)} of {len(keyed)} candidate rows re-checked, {'newest' if newest else 'oldest'} first "
            f"by time, from {window[0][1] if window else '-'} to {window[-1][1] if window else '-'}")
    return {"pairs": pairs, "coverage": coverage, "bounded": len(keyed) > start + len(window), "plan": found.plan,
            "span": span, "examined": len(window),
            "path": f"postings ({found.posted} posted, {found.dirty} edited, {found.uncovered} not yet posted)"}


def _snapshot(coverage: Mapping[str, Any], query: Mapping[str, Any]) -> str:
    """One captured generation plus the normalized words query, as the legacy reader's token hashes its own."""
    payload = {"schema_version": 2, "reader": "words", "query": dict(query),
               "generations": [{"name": pathlib.Path(str(row.get("path") or "")).name,
                                "first_line_sha256": str(row.get("first_line_sha256") or ""),
                                "size": int(row.get("size") or 0)}
                               for row in coverage.get("generations") or [] if isinstance(row, Mapping)],
               "durable_gap_ids": [str(value) for value in coverage.get("durable_gap_ids") or []]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def field_speaker(field: Field, row: Mapping[str, Any], author: Mapping[str, Any]) -> str:
    """The mind's label for who wrote ``field``: the row's author, ``task <id>`` for the asking or closing work,
    ``my human`` on a card, or ``host`` (the door maps ``field.speaker`` to its own outward words)."""
    if field.speaker == "author":
        return str(author.get("label") or "author not recorded")
    if field.speaker == "asker":
        task = str(row.get("task_id") or "")
        return f"task {task}" if task else "the asking task"
    if field.speaker == "closer":
        quiz = row.get("quiz") if isinstance(row.get("quiz"), Mapping) else {}
        task = str(quiz.get("closed_by_task_id") or "")
        return f"task {task}" if task else "the closing task"
    return SPEAKER_WORDS.get(field.speaker, field.speaker)


def _seals(root: pathlib.Path, wanted: List[Tuple[int, str]]) -> Tuple[Dict[Tuple[int, str], str], str]:
    """``({(room, row_sha256): page id}, gap)`` from the chronicle index as it stands, read-only: no import,
    no catch-up, no lock. The gap names why no seal was read on this path; ``""`` when they were."""
    from ouroboros.chronicle_store import ChronicleStore

    store = ChronicleStore(root)
    if not store.log_path.exists():
        return {}, "chronicle not activated: no page seals any row yet"
    db = store._current()  # the lock-free read path; None when no current read-only index is at hand
    if db is None:
        return {}, ("page seals were not read on this path (no current read-only chronicle index here); "
                    "memory_read of the row's room shows them")
    found: Dict[Tuple[int, str], str] = {}
    try:
        by_room: Dict[int, List[str]] = {}
        for room, sha in wanted:
            by_room.setdefault(room, []).append(sha)
        for room, shas in by_room.items():
            for at in range(0, len(shas), _CHUNK):
                chunk = shas[at:at + _CHUNK]
                for page_id, ref in db.execute(
                        f"SELECT page_id, row_ref FROM page_rows WHERE room=? AND row_ref IN ({','.join('?' * len(chunk))})",
                        (str(room), *chunk)):
                    found[(room, str(ref))] = str(page_id)
    finally:
        db.close()
    return found, ""


class Showing:
    """The rooms that show a row now, by the store's own projection (``memory_inventory.rooms_of_row`` over one
    ``membership_facts`` capture: the room rule plus a Project's founding owner message), read once per read and
    used for admission, labels, seals and reopen alike. An unreadable project registry is ``gap`` (the caller
    counts it) and each row then names the chat it was written in."""

    def __init__(self, root: pathlib.Path):
        from ouroboros.projects_registry import reserved_project_chat_ids

        self.facts, self.gap = None, ""
        try:
            reserved_project_chat_ids(root, strict=True)  # the registry must read; membership's own read is lenient
            from ouroboros.memory_inventory import membership_facts

            self.facts = membership_facts(root)
        except Exception as exc:
            self.gap = f"rooms_unreadable ({type(exc).__name__}): each row names the chat it was written in"
        self._known: Dict[str, List[int]] = {}

    def rooms(self, row: Mapping[str, Any]) -> List[int]:
        from ouroboros.chat_chain import row_address, source_row_id
        from ouroboros.memory_inventory import rooms_of_row, row_meta

        key = source_row_id(dict(row))
        if key not in self._known:
            self._known[key] = (sorted(int(room) for room in rooms_of_row(row_meta(row), self.facts))
                                if self.facts is not None else [int(row_address(row)["chat_id"])])
        return self._known[key]


def _authors(root: pathlib.Path, pairs: List[Tuple[int, Dict[str, Any]]], lineage: Mapping[str, Any]
             ) -> Tuple[Dict[int, Dict[str, Any]], str]:
    """Each page row's author as ``memory_read(rows=true)`` names it: the lineage epoch with the row's stream
    position, looked up once for the page."""
    from ouroboros import dialogue_positions as dp
    from ouroboros.dialogue_provenance import row_author

    gap, positions = "", {}
    if lineage.get("lineage_epoch") is not None:
        try:
            positions = {seq: found.pos for seq, found in dp.ordinals_of(root, "chat", [s for s, _r in pairs]).items()}
        except Exception as exc:
            gap = f"lineage_positions_unknown ({type(exc).__name__}): rows before the lineage epoch are named without it"
    plain = {key: value for key, value in lineage.items() if key == "owner_accounts"}
    out = {}
    for seq, row in pairs:
        pos = positions.get(seq)
        out[seq] = row_author(row, pos=pos, **lineage) if pos is not None else row_author(row, **plain)
    return out, gap


def label_authors(root: pathlib.Path, pairs: List[Tuple[int, Dict[str, Any]]]
                  ) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """The authors of the ``(seq, row)`` pairs handed in, named as ``search`` names its page rows (and
    ``memory_read(rows=true)`` names them): the places' owner accounts, the lineage epoch and each row's stream
    position. ``(authors by seq, notes)``; the notes say what could not be read."""
    lineage, note = _lineage(root)
    authors, gap = _authors(root, pairs, lineage)
    unread = [] if "owner_accounts" in lineage else [OWNER_ACCOUNTS_UNREAD]
    return authors, unread + [item for item in (note, gap) if item]


def _lineage(root: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    """``row_author`` keywords as ``chronicle_import.row_lineage`` gives them (the places' owner accounts, the
    activation receipt's lineage epoch and the task lookup), the receipt read on the lock-free index path only:
    this reader never catches the chronicle index up. ``(keywords, note)``."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_provenance import task_lineage_lookup

    keywords: Dict[str, Any] = {}
    try:
        from ouroboros.places import place_owner_accounts

        keywords["owner_accounts"] = place_owner_accounts(root)
    except Exception:
        log.debug("owner accounts unreadable for a words read", exc_info=True)
    store = ChronicleStore(root)
    if not store.log_path.exists():
        return keywords, ""
    db = store._current()
    if db is None:
        return keywords, ("lineage epoch not read on this path (no current read-only chronicle index): rows from "
                          "before it are named as their fields say")
    try:
        receipt = ChronicleStore._state(db, "activation") or {}
    finally:
        db.close()
    epoch = (receipt.get("metadata") or {}).get("lineage_epoch") if isinstance(receipt, dict) else None
    if isinstance(epoch, dict) and type(epoch.get("pos")) is int:
        keywords.update(lineage_epoch=epoch, lineage_lookup=task_lineage_lookup(root))
    return keywords, ""


def _reader(room: Optional[int], row: Mapping[str, Any], address: str) -> str:
    from ouroboros.chat_chain import context_address

    if room is None:  # no room shows it: its identity reads it
        return f"memory_read(source_ref={json.dumps(context_address(dict(row)), sort_keys=True)})"
    return f'memory_read(rows=true, room_id="{room}", from="{address}", to="{address}")'


_TOKEN = re.compile(r"[0-9a-f]{64}(:[0-9]+)?")
UNSTABLE_GAP = ("capture_unstable: the conversation changed during this read (an edited row); the count is what it "
                "saw; start again without snapshot or offset")
_REFUSALS = {
    "malformed": "CHAT_HISTORY_SNAPSHOT_INVALID: this snapshot was not issued by a words read; start again with "
                 "offset=0 and no snapshot.",
    "unstable": "CHAT_HISTORY_SNAPSHOT_CHANGED: the conversation changed during this read; no mixed page was "
                "returned; restart with offset=0 and no snapshot.",
    "changed": "CHAT_HISTORY_SNAPSHOT_CHANGED: the query, the archive/live generations or an edited row changed; "
               "no mixed page was returned; restart with offset=0 and no snapshot.",
}


def _refusal(kind: str) -> Dict[str, Any]:
    """A continuation this read cannot serve: ``malformed`` (no read issued it), ``unstable`` (the capture moved
    under it) or ``changed`` (another capture or query)."""
    return {"refused": _REFUSALS[kind], "refused_kind": kind}


def search(memory: Any, query: WordQuery, *, room: Optional[int], voice: str = "", order: str = "",
           count: int = 100, offset: int = 0, snapshot: str = "", filters: Mapping[str, str],
           visible_only: bool = False) -> Dict[str, Any]:
    """One page of a words read as data, for the two faces that print it (``words_history`` here, the door).

    Keys: ``refused`` (a snapshot sentence, else ``None``); ``hits`` (each: ``row``, ``seq``, ``sha256``,
    ``address`` (the 12-digit form), ``written_chat``, ``rooms`` (showing now), ``reader_room``, ``reader``,
    ``author``, ``api_door``, ``reply_to``, ``reply_unresolved``, ``matched``, ``fields`` (each ``name``,
    ``text``, ``speaker`` (``author|asker|owner|closer|host``), ``label``), ``seals`` (``[(room, page)]``));
    ``total`` (exact, else ``None``) and ``observed``; ``offset``, ``more``; ``gaps``, ``notes``, ``seal_gap``;
    ``snapshot`` (this page's token, ``None`` when the capture did not stabilize), ``beyond`` (the token of
    the next re-check window, else ``None``); ``span`` (the examined range of a windowed read); ``path``.
    ``visible_only`` (the neighbour door) keeps only rows a visible room shows now (a positive room id: Main, a
    Project, a place), resolved by the room rule before counting and paging; the mind's reads keep every row."""
    from ouroboros.chat_chain import format_address, row_address, source_row_id
    from ouroboros.dialogue_provenance import owner_speech, row_author
    from ouroboros.memory import chat_history_filter, normalized_chat_history_query

    root = pathlib.Path(memory.drive_root)
    exact = chat_history_filter(filters, "")
    notes: List[str] = []
    gaps_extra: List[str] = []
    lineage, lineage_note = _lineage(root)
    if lineage_note:
        notes.append(lineage_note)
    if "owner_accounts" not in lineage:  # no place row reads as my human's then; said, never guessed
        notes.append(OWNER_ACCOUNTS_UNREAD)
        if voice == "owner":
            gaps_extra.append("owner_accounts_unreadable: place rows of my human not confirmed")
    accounts = lineage.get("owner_accounts")
    token = str(snapshot or "").strip().lower()
    requested, colon, start_text = token.partition(":")
    if token and (not _TOKEN.fullmatch(token) or (colon and int(start_text) == 0)):
        return {**_refusal("malformed")}
    start = int(start_text) if colon else 0
    revision = _text_revision(root)
    judged: Dict[int, List[str]] = {}

    def judge(row: Dict[str, Any]) -> bool:
        if not exact(row):
            return False
        if voice == "owner" and not owner_speech(row, row_author(row, owner_accounts=accounts)):
            return False  # a row selector; each field keeps its speaker
        matched = match_fields(query, [field for field in fields_of(row) if field.words])
        if matched:
            judged[id(row)] = matched
        return bool(matched)

    newest = order != "oldest"
    read = _posted(memory, query, newest=newest, start=start) if room is None else None
    if (read or {}).get("malformed") or (start and read is None):
        return _refusal("malformed")  # a window suffix this read never issued (a scan issues none)
    if read is not None:
        read["pairs"] = [(seq, row) for seq, row in read["pairs"] if judge(row)]
    else:
        read = _scan(memory, room, judge)
    coverage = read["coverage"]
    seen, hits_all = set(), []
    for seq, row in sorted(read["pairs"], key=lambda pair: time_key(pair[1], pair[0]), reverse=newest):
        identity = (row_address(row)["chat_id"], str(row.get("ts") or ""), source_row_id(row))
        if identity not in seen:  # one source row, whatever its fields or physical copies
            seen.add(identity)
            hits_all.append((seq, row))
    showing = Showing(root)  # one projection of the rooms for this read: admission, labels, seals, readers
    if visible_only:  # the hidden partition (0) and synthetic traffic are no room a person reads
        hits_all = [(seq, row) for seq, row in hits_all if any(chat > 0 for chat in showing.rooms(row))]
    moved = _text_revision(root) != revision
    stable = bool(coverage.get("snapshot_stable")) and not moved
    windowed = read["bounded"] or bool(start)
    normalized = {**normalized_chat_history_query(filters, ""), "words": list(query.items),
                  "room_id": "" if room is None else str(room), "voice": voice, "order": order or "newest",
                  "chat_text_revision": revision, "plan": read["plan"] if windowed else "",
                  **({"visible_only": True} if visible_only else {})}
    current = _snapshot(coverage, normalized)
    if requested and (not stable or requested != current):
        return _refusal("unstable" if not stable else "changed")
    result: Dict[str, Any] = {"refused": None, "refused_kind": None, "path": read["path"], "span": "",
                              "beyond": None}
    gaps = [f"{gap.get('kind') or 'unknown'}" + (f" ({gap['detail']})" if gap.get("detail") else "")
            for gap in coverage.get("gaps") or []] + gaps_extra
    if showing.gap:
        gaps.append(showing.gap)
    if moved:  # an edit landed while this read ran: its count is what it saw, never exact
        gaps.append(UNSTABLE_GAP)
    if windowed:
        # The door's answer never carries the time or address of a row it does not serve: its window says counts.
        result["span"] = read["span"] if not visible_only else read["span"].split(", from ")[0]
        if read["bounded"] and stable:
            result["beyond"] = f"{current}:{start + read['examined']}"
        gaps.append(f"recheck_bound: {result['span']}; " + (
            f"the rest was not examined: continue beyond with offset=0, snapshot={result['beyond']}"
            if result["beyond"] else "the rest was not examined" if read["bounded"] else "this is the last window")
            + (f"; earlier windows held {start} candidates" if start else ""))
    offset, count = max(0, int(offset)), max(1, int(count))
    page = hits_all[offset:offset + count]
    rooms_of = {seq: showing.rooms(row) for seq, row in page}
    authors, lineage_gap = _authors(root, page, lineage)
    if lineage_gap:
        notes.append(lineage_gap)
    wanted = [(chat, source_row_id(row)) for seq, row in page
              for chat in {*rooms_of[seq], *([room] if room is not None else [])}]
    seals, seal_gap = _seals(root, wanted) if page else ({}, "")
    hits = []
    for seq, row in page:
        address = format_address(row_address(row))
        written = row_address(row)["chat_id"]
        rooms = rooms_of[seq]
        reader_room = room if room is not None else (written if written in rooms else (rooms[0] if rooms else None))
        author = authors[seq]
        reply = row.get("reply_to") if isinstance(row.get("reply_to"), Mapping) else {}
        surface = row.get("client_surface") if isinstance(row.get("client_surface"), Mapping) else {}
        fields = fields_of(row)
        sha = source_row_id(row)
        hits.append({
            "row": row, "seq": seq, "sha256": sha, "address": address, "written_chat": written, "rooms": rooms,
            "reader_room": reader_room, "reader": _reader(reader_room, row, address), "author": author,
            "api_door": surface.get("channel") == "api_command",
            "reply_to": str(reply.get("address") or ""), "reply_unresolved": bool(reply.get("unresolved")),
            "matched": judged.get(id(row), []),
            "fields": [{"name": field.name, "text": field.text, "speaker": field.speaker,
                        "label": field_speaker(field, row, author)} for field in fields],
            "seals": [(chat, seals[(chat, sha)]) for chat in sorted({*rooms, *([room] if room is not None else [])})
                      if (chat, sha) in seals]})
    result.update(hits=hits, total=None if gaps else len(hits_all), observed=len(hits_all), offset=offset,
                  more=max(0, len(hits_all) - offset - len(page)), gaps=gaps, notes=notes, seal_gap=seal_gap,
                  snapshot=(f"{current}:{start}" if start else current) if stable else None)
    return result


def _line(root: pathlib.Path, hit: Mapping[str, Any], query: WordQuery, rooms: Any) -> str:
    from ouroboros.dialogue_provenance import dialogue_provenance

    row = hit["row"]
    where = ", ".join(rooms.label({"chat_id": chat}) for chat in hit["rooms"]) or "shown in no room"
    if hit["rooms"] and hit["written_chat"] not in hit["rooms"]:
        where += f" (written in chat {hit['written_chat']})"
    who = str(hit["author"].get("label") or "author not recorded")
    provenance = dialogue_provenance(row)
    if provenance and provenance not in who:
        who += f" [{provenance}]"
    if hit["api_door"]:
        who += f"; {API_DOOR_NOTE}"
    head = f"{hit['address']} · {where} · {row.get('ts') or 'time not recorded'} · {who} · matched " + \
        ", ".join(hit["matched"])
    if hit["reply_to"]:
        head += f" · replies to {hit['reply_to']}" + (" (not resolved)" if hit["reply_unresolved"] else "")
    for chat, page in hit["seals"]:
        head += f' · sealed by page {page} in room {chat}: memory_read(node_id="{page}")'
    head += f" · read: {hit['reader']}"
    fields = hit["fields"]
    card = len(fields) > 1 or (fields and fields[0]["name"] != "text")
    shown = fields if card else [field for field in fields if field["name"] in hit["matched"]]
    return head + "".join(f"\n  {field['name']} ({field['label']}): {excerpt(query, field['text'], EXCERPT_CHARS)}"
                          for field in shown)


def words_history(memory: Any, query: WordQuery, *, room: Optional[int], voice: str = "", order: str = "",
                  count: int = 100, offset: int = 0, snapshot: str = "", filters: Mapping[str, str]) -> str:
    """One page of rows whose typed fields satisfy ``query``, as text (``search`` rendered)."""
    from ouroboros.dialogue_provenance import RoomLabelResolver

    root = pathlib.Path(memory.drive_root)
    try:
        found = search(memory, query, room=room, voice=voice, order=order, count=count, offset=offset,
                       snapshot=snapshot, filters=filters)
    except Exception as exc:  # the legacy reader's sentence for a store it cannot read
        return f"(error reading history: {exc})"
    if found["refused"]:
        return found["refused"]
    hits, gaps = found["hits"], found["gaps"]
    counted = f"total={found['total']}" if found["total"] is not None else f"observed={found['observed']}; total_unknown"
    header = (f"Words {json.dumps(list(query.items), ensure_ascii=False)} in "
              + ("every room" if room is None else f"room {room}")
              + (", my human's rows only" if voice == "owner" else "")
              + f", {'oldest' if order == 'oldest' else 'newest'} first: {counted}; "
              f"showing {len(hits)} from offset={found['offset']}; {found['more']} more.")
    if found["snapshot"]:
        header += f" Continue with offset={found['offset'] + len(hits)}, snapshot={found['snapshot']}."
    else:
        header += " Snapshot unavailable because the capture did not stabilize."
    if gaps:
        header += " Gaps: " + "; ".join(gaps) + "."
    if found["seal_gap"]:
        header += f" Seals: {found['seal_gap']}."
    for note in found["notes"]:
        header += f" Note: {note}."
    header += f" Read by {found['path']}."
    if not hits:
        return header + ("\n\n(no rows whose words match)" if not found["observed"] else "")
    rooms = RoomLabelResolver(root)
    return header + "\n\n" + "\n".join(_line(root, hit, query, rooms) for hit in hits)
