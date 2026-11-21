"""``chat_history(words=…)``: words over the conversation, read only when the mind calls (consensus K4).

The captured set is the one ``Memory.read_chat_generations`` captures for the legacy reader (the frozen
upper bound, its import and invalid-interval gaps, a room's appearance gaps, the durable gap blocks), so
both filters see the same rows and name the same gaps. Its candidates are every row of that set the exact
filters keep (``memory._chat_history_filter``; A2A traffic never). Each candidate is re-checked by the one
predicate (``dialogue_words``) on its typed fields, an alternative inside one field. ``RECHECK_BOUND`` is the
one physical bound: past it the read returns the examined range, never a block. The word postings, when
they exist, hand in fewer candidates through the same seam (``_candidates``); the predicate stays the judge.

No relevance order and no host choice of a row: ``order`` (the caller's direction of time) and the page
are all. ``voice=owner`` keeps the rows ``dialogue_provenance.owner_speech`` already calls my human's words
and, on a card, only the fields my human wrote. A row stamped by the API door (``client_surface.channel ==
"api_command"``) says that its sender is not recorded; any other row claims nothing about its route.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.dialogue_words import Field, WordQuery, fields_of, match_fields, normalize, parse

# Candidate rows one call re-checks at most: a bound on one read's time and memory (a 100k-row store
# re-checks in seconds; REPORT of C06/P1), never a choice of which rows matter. Met, it is disclosed.
RECHECK_BOUND = 500_000
EXCERPT_CHARS = 240  # the shown part of one field around its first match; the row's reader holds it whole
ORDERS = ("newest", "oldest")
VOICES = ("owner",)
API_DOOR_NOTE = "API door, sender not recorded"
_SPEAKER_WORDS = {"owner": "my human", "host": "host"}
# The owner-facing sentence of ``chat_history``'s ``words`` (tools/control.py), beside the reader it describes.
WORDS_DESCRIPTION = (
    "Instead of search: alternatives, any one may match. Each is words that must all occur, in any order, as word "
    "prefixes within one field of one row (message text, a card's question, an offered option, the comment on an "
    "answer); a part in double quotes is an exact word or phrase. Case, ё/е and Unicode forms do not matter; no "
    "ranking. Each line names the row's address, room, date, speaker, matched field and its readers.")


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

    ``words`` and the legacy ``search`` exclude each other; ``room_id``, ``voice`` and ``order`` go with
    ``words`` only, so a call without ``words`` keeps the legacy reader's contract exactly. With ``words`` the
    exact ``filters`` are checked here too (a date that is not ISO-8601), before anything is read."""
    if words is None or words == []:
        if words == []:
            parse(words)  # the empty query's own message
        given = [name for name, value in (("room_id", room_id), ("voice", voice), ("order", order))
                 if str(value if value is not None else "").strip()]
        if given:
            raise ValueError(f"{', '.join(given)} {'go' if len(given) > 1 else 'goes'} with words=[…]; "
                             "the substring search takes none of them")
        return None, None
    if str(search or "").strip():
        raise ValueError("words and search exclude each other: words=[…] matches word prefixes within one field, "
                         "search=\"…\" keeps the old substring filter; give one")
    if str(voice or "") not in ("", *VOICES):
        raise ValueError(f"voice is {' or '.join(VOICES)} (omit it for every speaker); got {voice!r}")
    if str(order or "") not in ("", *ORDERS):
        raise ValueError(f"order is newest or oldest (default newest); got {order!r}")
    query, room = parse(words), _room(room_id)
    from ouroboros.memory import _chat_history_filter

    _chat_history_filter(filters or {}, "")
    return query, room


def _candidates(memory: Any, room: Optional[int], keep: Callable[[Dict[str, Any]], bool]):
    """The captured set's rows that ``keep`` admits, with the legacy reader's coverage: the scan path.
    A later source (the word postings) narrows what reaches ``keep``; ``keep`` judges every row it sees."""
    return memory.read_chat_generations(exclude_a2a=True, predicate=keep, room_id=room)


def _snapshot(coverage: Mapping[str, Any], query: Mapping[str, Any]) -> str:
    """One captured generation plus the normalized words query, as the legacy reader's token hashes its own."""
    payload = {"schema_version": 1, "reader": "words", "query": dict(query),
               "generations": [{"name": pathlib.Path(str(row.get("path") or "")).name,
                                "first_line_sha256": str(row.get("first_line_sha256") or ""),
                                "size": int(row.get("size") or 0)}
                               for row in coverage.get("generations") or [] if isinstance(row, Mapping)],
               "durable_gap_ids": [str(value) for value in coverage.get("durable_gap_ids") or []]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _excerpt(text: str, query: WordQuery) -> str:
    """The field's words around its first matching token, cut at ``EXCERPT_CHARS`` and saying so."""
    text = " ".join(str(text).split())
    if len(text) <= EXCERPT_CHARS:
        return text
    start = 0
    folded = normalize(text)
    if len(folded) == len(text):  # offsets carry over only when normalizing kept the length
        firsts = [folded.find(part) for alternative in query.alternatives
                  for part in (*alternative.prefixes, *(" ".join(phrase) for phrase in alternative.phrases))]
        found = [position for position in firsts if position >= 0]
        start = max(0, min(found) - EXCERPT_CHARS // 4) if found else 0
    cut = text[start:start + EXCERPT_CHARS]
    return ("…" if start else "") + cut + ("…" if start + EXCERPT_CHARS < len(text) else "") + \
        f" [{len(text)} chars in full]"


def _speaker(field: Field, row: Mapping[str, Any], author: Mapping[str, Any]) -> str:
    if field.speaker == "author":
        return str(author.get("label") or "author not recorded")
    if field.speaker == "asker":
        task = str(row.get("task_id") or "")
        return f"task {task}" if task else "the asking task"
    if field.speaker == "closer":
        quiz = row.get("quiz") if isinstance(row.get("quiz"), Mapping) else {}
        task = str(quiz.get("closed_by_task_id") or "")
        return f"task {task}" if task else "the closing task"
    return _SPEAKER_WORDS.get(field.speaker, field.speaker)


def _seals(root: pathlib.Path, wanted: Sequence[Tuple[int, str]]) -> Tuple[Dict[Tuple[int, str], str], str]:
    """``({(room, row_sha256): page id}, gap)`` from the chronicle index as it stands, read-only: no import,
    no catch-up, no lock (a passive read). The gap names why no seal could be read; ``""`` when they were."""
    from ouroboros.chronicle_store import ChronicleStore

    store = ChronicleStore(root)
    if not store.log_path.exists():
        return {}, "chronicle not activated: no page seals any row yet"
    db = store._current()  # the lock-free read path; None while the index is behind its journal
    if db is None:
        return {}, "page seals not read: the chronicle index is behind its journal (any memory_read catches it up)"
    found: Dict[Tuple[int, str], str] = {}
    try:
        by_room: Dict[int, List[str]] = {}
        for room, sha in wanted:
            by_room.setdefault(room, []).append(sha)
        for room, shas in by_room.items():
            marks = ",".join("?" * len(shas))
            for page_id, ref in db.execute(f"SELECT page_id, row_ref FROM page_rows WHERE room=? AND row_ref IN ({marks})",
                                           (str(room), *shas)):
                found[(room, str(ref))] = str(page_id)
    finally:
        db.close()
    return found, ""


def _line(root: pathlib.Path, row: Dict[str, Any], hit: Mapping[str, Any], query: WordQuery, *, rooms: Any,
          shown_in: Optional[int], seals: Mapping[Tuple[int, str], str]) -> str:
    from ouroboros.chat_chain import format_address, row_address, source_row_id
    from ouroboros.room_appearances import frame_rooms

    address = format_address(row_address(row))
    author, fields, matched = hit["author"], hit["fields"], hit["matched"]
    chat = int(row.get("chat_id", 1)) if str(row.get("chat_id", 1)).lstrip("-").isdigit() else None
    where = rooms.label(row)
    if shown_in is not None and chat is not None and shown_in != chat:
        where += f", shown in room {shown_in}"
    also = frame_rooms(root, row, stamped_only=True).get("also_rooms") or []
    if also:
        where += ", also shown in room " + ", ".join(str(room) for room in also)
    from ouroboros.dialogue_provenance import dialogue_provenance

    who = str(author.get("label") or "author not recorded")
    provenance = dialogue_provenance(row)
    if provenance and provenance not in who:
        who += f" [{provenance}]"
    surface = row.get("client_surface") if isinstance(row.get("client_surface"), Mapping) else {}
    if surface.get("channel") == "api_command":
        who += f"; {API_DOOR_NOTE}"
    head = f"{address} · {where} · {row.get('ts') or 'time not recorded'} · {who}"
    head += " · matched " + ", ".join(matched)
    reply = row.get("reply_to") if isinstance(row.get("reply_to"), Mapping) else {}
    if str(reply.get("address") or ""):
        head += f" · replies to {reply['address']}" + (" (not resolved)" if reply.get("unresolved") else "")
    sealed = [(room, seals.get((room, source_row_id(row)))) for room in sorted({room for room in (chat, shown_in, *also)
                                                                                if room is not None})]
    for room, page in sealed:
        if page:
            head += f" · sealed by page {page} in room {room}: memory_read(node_id=\"{page}\")"
    read_room = shown_in if shown_in is not None else chat
    head += f" · read: memory_read(rows=true, room_id=\"{read_room}\", from=\"{address}\", to=\"{address}\")"
    card = len(fields) > 1 or (fields and fields[0].name != "text")
    shown = fields if card else [field for field in fields if field.name in matched]
    return head + "".join(f"\n  {field.name} ({_speaker(field, row, author)}): {_excerpt(field.text, query)}"
                          for field in shown)


def words_history(memory: Any, query: WordQuery, *, room: Optional[int], voice: str = "", order: str = "",
                  count: int = 100, offset: int = 0, snapshot: str = "", filters: Mapping[str, str]) -> str:
    """One page of rows whose typed fields satisfy ``query``, as text (see the module docstring)."""
    from ouroboros.dialogue_provenance import (RoomLabelResolver, in_reading_order, owner_speech,
                                               rooms_read_by_time, row_author)
    from ouroboros.memory import _chat_history_filter, _normalized_chat_history_query

    root = pathlib.Path(memory.drive_root)
    exact = _chat_history_filter(filters, "")
    notes: List[str] = []  # gaps of this read; ``said`` qualifies labels without touching the count
    said = ""
    try:
        from ouroboros.places import place_owner_accounts

        accounts = place_owner_accounts(root)
    except Exception as exc:  # no place row reads as my human's then; said, never guessed
        accounts = None
        said = f"owner accounts unreadable ({type(exc).__name__}): no place row is labelled as my human's"
        if voice == "owner":
            notes.append(f"owner_accounts_unreadable ({type(exc).__name__}): place rows of my human not confirmed")
    hits: Dict[int, Dict[str, Any]] = {}
    examined = {"rows": 0, "last": None, "bounded": False}

    def keep(row: Dict[str, Any]) -> bool:
        if not exact(row):
            return False
        if examined["rows"] >= RECHECK_BOUND:  # a candidate past the bound: the range ends before it
            examined["bounded"] = True
            return False
        examined["rows"] += 1
        examined["last"] = row
        author = row_author(row, owner_accounts=accounts)
        fields = fields_of(row)
        searched = [field for field in fields if field.words]
        if voice == "owner":  # my human's rows, and on a card only what my human wrote; the rest stays shown
            if not owner_speech(row, author):
                return False
            searched = [field for field in searched if field.speaker in ("author", "owner")]
        matched = match_fields(query, searched)
        if matched:
            hits[id(row)] = {"author": author, "fields": fields, "matched": matched}
        return bool(matched)

    entries, coverage = _candidates(memory, room, keep)
    entries = in_reading_order(entries, rooms_read_by_time(root, entries))
    from ouroboros.chat_chain import format_address, row_address

    seen, unique = set(), []
    for row in entries:  # a source row is one line whatever its fields or physical copies
        address = format_address(row_address(row))
        if address not in seen:
            seen.add(address)
            unique.append(row)
    if order != "oldest":
        unique.reverse()
    normalized = {**_normalized_chat_history_query(filters, ""), "words": list(query.items),
                  "room_id": "" if room is None else str(room), "voice": voice, "order": order or "newest"}
    current = _snapshot(coverage, normalized)
    requested = str(snapshot or "").strip().lower()
    if requested and (not coverage.get("snapshot_stable") or requested != current):
        return ("CHAT_HISTORY_SNAPSHOT_CHANGED: the query or archive/live generations "
                "changed; no mixed page was returned; restart with offset=0 and no snapshot.")
    gaps = [f"{gap.get('kind') or 'unknown'}" + (f" ({gap['detail']})" if gap.get("detail") else "")
            for gap in coverage.get("gaps") or []] + notes
    if examined["bounded"] and examined["last"] is not None:
        last = examined["last"]
        gaps.append(f"recheck_bound: {RECHECK_BOUND} candidate rows re-checked in stored order, through "
                    f"{format_address(row_address(last))} ({last.get('ts')}); later rows were not examined: "
                    "narrow with room_id or date_from/date_to")
    offset, count = max(0, int(offset)), max(1, int(count))
    page = unique[offset:offset + count]
    rooms = RoomLabelResolver(root)
    wanted = []
    from ouroboros.chat_chain import source_row_id
    from ouroboros.room_appearances import frame_rooms

    for row in page:
        sha = source_row_id(row)
        for chat in {row.get("chat_id", 1), room, *(frame_rooms(root, row, stamped_only=True).get("also_rooms") or [])}:
            if chat is not None and str(chat).lstrip("-").isdigit():
                wanted.append((int(chat), sha))
    seals, seal_gap = _seals(root, wanted) if page else ({}, "")
    direction = "newest first" if order != "oldest" else "oldest first"
    scope = "every room" if room is None else f"room {room}"
    counted = (f"total={len(unique)}" if not gaps else
               f"observed={len(unique)}; total_unknown")
    header = (f"Words {json.dumps(list(query.items), ensure_ascii=False)} in {scope}"
              + (", my human's words only" if voice == "owner" else "") + f", {direction}: {counted}; "
              f"showing {len(page)} from offset={offset}; {max(0, len(unique) - offset - len(page))} more.")
    if coverage.get("snapshot_stable"):
        header += f" Continue with offset={offset + len(page)}, snapshot={current}."
    else:
        header += " Snapshot unavailable because the generation capture did not stabilize."
    if gaps:
        header += " Gaps: " + "; ".join(gaps) + "."
    if seal_gap:
        header += f" Seals: {seal_gap}."
    if said and voice != "owner":
        header += f" Note: {said}."
    if not page:
        return header + ("\n\n(no rows whose words match)" if not unique else "")
    return header + "\n\n" + "\n".join(_line(root, row, hits[id(row)], query, rooms=rooms, shown_in=room,
                                             seals=seals) for row in page)
