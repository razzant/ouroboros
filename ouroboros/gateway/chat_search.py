"""Search of one room's saved messages: ``GET /api/chat/search`` (DESIGN "Search in a chat").

A search reads the room the way its history pages do, with the same authorities: the room's
context and rule (``history._project_history_context``, ``history._make_thread_filter``), its
lens (``history_paging.room_lens``) and its stream reader (``history_paging.history_source``: a
room holding imported earlier messages reads in time order). One request reads at most
``_SCAN_ROWS`` of the room's stored chat rows below its cut, newest first, through that reader's
bounded ``older`` read, and answers the saved messages among them whose words contain the query:
a literal, case-insensitive (Unicode case folding) substring of the message's text with runs of
white space read as one space -- never a pattern, a row's metadata, attachment or file contents,
a task card's content (a host placement, a child's words) or task narration (progress rows are read
only to name a legacy child, never searched). A speaker's lineage is the room rule's indexed facts
(``RoomResolver``), so no task result, queue snapshot, events or annotations journal is read. The
room's authority is read strictly, as an exact row's landing reads it.

The cursor binds the room, its view, the query and the frozen end of the chat stream, and names
the cut the next read continues below (a stored coordinate, or for a time-ordered room the row
key ``history_page_tokens`` would carry), so a continuation reads the same set and a message
that arrived since is not mixed into it. Nothing is indexed and nothing is written. A read that
reached the archive import's frontier says older history is still importing (``importing``) and
a range holding lines the import could not store names them (``gaps``): an unread range is never
"no match", nor is a room whose registry, bindings or lineage cannot be read (unavailable, 503). A
page with no match keeps its continuation; only a chain that reached the room's beginning with no
gap has searched all of it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import pathlib
import unicodedata
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import Response

from ouroboros import dialogue_store as ds
from ouroboros.chat_references import project_answer_text
from ouroboros.chat_uploads import attachment_view
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.gateway.history import _make_thread_filter, _project_history_context, _stored_chat_id
from ouroboros.gateway.history_paging import (
    _CHRONOLOGICAL, _PROBE_ROWS, HistoryCursorError, encode_cursor, history_source, message_elsewhere, room_lens,
    room_member_predicate, room_view_fingerprint,
)
from ouroboros.store import StoreError, StoreMigrating
from ouroboros.subagent_messages import is_conversation_card_content, is_task_card_message, subagent_message_meta

# The stored chat rows one request reads, at most: the cursor carries the search on, so this bounds
# the request's latency (and its decoding), never what the room's search can reach.
_SCAN_ROWS = 2000
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50
_MAX_QUERY_CHARS = 200
# A match's snippet: about this many characters of the message's words, starting a little before the
# first match on a word boundary; a cut is shown with an ellipsis.
_SNIPPET_CHARS = 160
_LEAD_CHARS = 48
_CURSOR_CHARS = 1024
# Stored chat rows that are no message of the conversation: evidence for cards (an answer or closure
# of a question, a task's terminal summary, a skill review's row) and the routing options' grounding.
_NOT_MESSAGES = frozenset({"routing_options", "quiz_answer", "quiz_closed", "task_summary", "skill_review"})


def _words(text: Any) -> str:
    """A message's words as searched and shown: NFC, every run of white space one space."""
    return unicodedata.normalize("NFC", " ".join(str(text or "").split()))


def normalize_query(raw: Any) -> str:
    query = _words(raw)
    if not query or len(query) > _MAX_QUERY_CHARS:
        raise HistoryCursorError("search_query_invalid", 400)
    return query


def _digest(folded: str) -> str:
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()[:16]


def _saved_message(row_matches_thread, child):
    """Whether a stored chat row is a message this room shows as one (``history._collect_chat_rows``):
    a member of the room with a known author, not A2A traffic, card evidence or the host's placeholder
    for an owner message without words, nor a row whose message lives elsewhere -- Main's copy of a
    Project's question, open or settled, and a work's portal (``history_paging.message_elsewhere``, as
    unread reads them; a legacy standalone notice stays a message) -- and no task card's content -- a
    host placement or a child's words, by the rule history and unread share
    (``is_conversation_card_content``); ``child`` reads the lineage a legacy row's task recovers."""
    def message(entry):
        kind = str(entry.get("type") or "")
        if kind in _NOT_MESSAGES or entry.get("text_placeholder") is True or is_a2a_chat_id(entry.get("chat_id", 1)):
            return False
        if entry.get("role") != "system" and str(entry.get("direction") or "").lower() not in ("in", "out", "system"):
            return False
        return (bool(row_matches_thread(_stored_chat_id(entry.get("chat_id"), 1), entry))
                and not message_elsewhere(entry, row_matches_thread)
                and not is_conversation_card_content(entry, kind, child))
    return message


def _legacy_children(rooms, found) -> set:
    """The found messages' speakers whose role an older body kept only in its task's ``metadata`` (no typed
    lineage): one statement over those tasks, the reader message dating uses (``room_messages._legacy_roles``).
    An unanswered read names none: those words stay matches, as before a child's card is known."""
    from ouroboros.room_messages import _legacy_roles

    untyped = {str(entry["task_id"]) for entry in found
               if entry.get("task_id") and not rooms.lineage(entry["task_id"]).get("delegation_role")}
    roles = (_legacy_roles(rooms, untyped) if untyped else None) or {}
    return {task_id for task_id, role in roles.items() if is_task_card_message(role)}


def _page_children(data_dir, rows, found, member, *, beginning: bool, recent: bool) -> set:
    """The children this page itself names, as ``history_paging.latest_arrival``'s ``known_children``: the
    lineage of the chat rows it read, and of the room's progress rows naming a found message's task as their
    child (one indexed probe of those tasks, at most ``_PROBE_ROWS`` rows, oldest first) written within the
    page's span -- from its oldest row read, or the room's beginning, to its newest, or the present on a first
    page. A child named only outside the page stays unknown there: its words stay a match, as that page of the
    feed shows them."""
    named = {meta["subagent_task_id"] for row in rows if (meta := subagent_message_meta(row))}
    tasks = {str(entry.get("task_id") or "") for entry in found} - named - {""}
    if not tasks:
        return named
    times = sorted(str(row.get("ts") or "") for row in rows)  # history orders by the stored ``ts`` string too
    lower, upper = "" if beginning else times[0], None if recent else times[-1]
    for row in ds.probe_rows(data_dir, "progress", [("subagent_task_id", task) for task in sorted(tasks)],
                             limit=_PROBE_ROWS, newest_first=False):
        entry, ts = row.row, str(row.row.get("ts") or "")
        meta = subagent_message_meta(entry)
        if (meta.get("subagent_task_id") in tasks and lower <= ts and (upper is None or ts <= upper)
                and member(entry, True)):
            named.add(meta["subagent_task_id"])
    return named


def snippet(text: Any, folded_query: str) -> List[Dict[str, Any]]:
    """The message's words around its first match as runs, ``match`` marking every occurrence in the
    window. Folding is per character (``str.casefold`` has no context), so a folded match maps back
    to the characters it came from even where folding changes a length (ß, İ)."""
    words = _words(text)
    owner: List[int] = []
    for index, char in enumerate(words):
        owner.extend([index] * len(char.casefold()))
    folded = words.casefold()
    found, at = [], folded.find(folded_query)
    while at >= 0 and folded_query:
        begin, stop = owner[at], owner[at + len(folded_query) - 1] + 1
        if found and begin < found[-1][1]:
            found[-1] = (found[-1][0], max(stop, found[-1][1]))
        else:
            found.append((begin, stop))
        at = folded.find(folded_query, at + len(folded_query))
    if not found:
        return [{"text": words[:_SNIPPET_CHARS], "match": False}]
    first, last = found[0]
    start = max(0, first - _LEAD_CHARS)
    if start:  # begin on a word: the space before it, if the lead holds one
        space = words.find(" ", start, first)
        start = space + 1 if space >= 0 else start
    end = max(last, min(len(words), start + _SNIPPET_CHARS))
    if end < len(words):
        space = words.rfind(" ", last, end)
        end = space if space > last else end
    parts, cursor = [], start
    for begin, stop in found:
        if begin < start or stop > end:
            continue
        if begin > cursor:
            parts.append({"text": words[cursor:begin], "match": False})
        parts.append({"text": words[begin:stop], "match": True})
        cursor = stop
    if cursor < end:
        parts.append({"text": words[cursor:end], "match": False})
    if start:
        parts.insert(0, {"text": "…", "match": False})
    if end < len(words):
        parts.append({"text": "…", "match": False})
    return parts


def _message_text(entry) -> str:
    """Words the bubble displays. Mirror chat_attachments.js::attachmentCaption, without reading files:
    only an incoming web-composed exact generated tail is hidden, using the server's rendered names.
    Other captions and authored lookalikes stay intact; the canonical row/address is never changed.
    """
    text = project_answer_text(entry) or str(entry.get("text") or "")
    refs = entry.get("attachments")
    if (entry.get("direction") != "in" or entry.get("role") == "system" or entry.get("source") != "web"
            or not isinstance(refs, list) or not refs):
        return text
    names = [attachment_view(ref)["name"] for ref in refs]
    tail = "\n".join([f"[Attached file: {name}]" for name in names[:25]]
                     + ([f"[{len(names) - 25} more attached files]"] if len(names) > 25 else []))
    if text == tail:
        return ""
    return text[:-len(tail) - 2] if text.endswith("\n\n" + tail) else text


def _match(entry, folded_query: str) -> Dict[str, Any]:
    direction = str(entry.get("direction") or "").lower()
    answer = project_answer_text(entry)
    role = ("assistant" if answer else "system" if entry.get("role") == "system"
            else {"in": "user", "out": "assistant"}.get(direction, "system"))
    found = {"history_id": entry["history_id"], "row_address": entry["row_address"], "role": role,
             "ts": str(entry.get("ts") or ""), "snippet": snippet(_message_text(entry), folded_query)}
    for key in ("sender_label", "task_id"):
        if entry.get(key):
            found[key] = str(entry[key])
    return found


def _decode_cursor(value: str, thread_id: int, view: str, digest: str, chrono: bool) -> Dict[str, Any]:
    try:
        if len(value) > _CURSOR_CHARS:
            raise ValueError
        cursor = json.loads(base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True))
        if not isinstance(cursor, dict) or cursor.get("v") != 1 or cursor.get("kind") != "search":
            raise ValueError
        if any(type(cursor.get(key)) is not int or cursor[key] <= 0 for key in ("upper", "before")):
            raise ValueError
        cut = cursor.setdefault("cut", None)
        if not chrono and cursor["before"] > cursor["upper"]:
            raise ValueError  # a physical cut lies at or below the frozen end it was read under
        if cut is not None and not (isinstance(cut, list) and len(cut) == 3 and type(cut[0]) is str
                                    and (cut[1] is None or type(cut[1]) is int) and type(cut[2]) is int):
            raise ValueError
    except (ValueError, TypeError):
        raise HistoryCursorError("search_cursor_invalid", 400) from None
    if cursor.get("q") != digest:
        raise HistoryCursorError("search_cursor_invalid", 400)  # a cursor continues the search it began
    if cursor.get("chat_id") != thread_id or cursor.get("view") != view or (cursor["cut"] is not None) != chrono:
        raise HistoryCursorError("search_view_changed")
    return cursor


def search_room(data_dir: pathlib.Path, thread_id: int, raw_query: Any, cursor: Optional[str] = None,
                limit: int = _DEFAULT_LIMIT) -> Dict[str, Any]:
    """One page of the room's search (``ChatSearchResponse``), newest match first, in the worker."""
    from ouroboros.room_appearances import read_gaps

    query = normalize_query(raw_query)
    folded = query.casefold()
    digest = _digest(folded)
    # Strict, as an exact row's landing reads it: "no match" is a claim about the whole room, so a registry, a binding
    # or a row's lineage the rule cannot read makes the search unavailable, never a room read without the rows it moves.
    project_chat_ids, source_refs, bindings, rooms, _classified = _project_history_context(
        data_dir, thread_id, strict=True)
    row_matches_thread = _make_thread_filter(thread_id, project_chat_ids, source_refs, rooms)
    lens = room_lens(data_dir, thread_id, project_chat_ids, source_refs, bindings, strict=True)  # its lineage too
    chrono = bool(lens["chat"].chrono)
    view = room_view_fingerprint(thread_id, project_chat_ids, source_refs, bindings) + (_CHRONOLOGICAL if chrono else "")
    state = _decode_cursor(cursor, thread_id, view, digest, chrono) if cursor else None
    reader = history_source(data_dir, "chat", lens["chat"], state["upper"] if state else None)
    reader.warm = rooms.warm  # the room rule's indexed lineage of every row read, before any is counted
    top = state["before"] if state else reader.horizon
    if chrono:
        reader.keys = {"before": state["cut"]} if state else {}
    # A speaker's typed lineage as that read left it: a lineage question opens no task result (its effective
    # projection would read the queue snapshot and the events journal for a task stored as running).
    is_message, hits = _saved_message(row_matches_thread, lambda entry: is_task_card_message(
        rooms.lineage(entry.get("task_id")))), set()

    def counts(entry):  # the words first (cheap), then the room rule
        if folded in _words(_message_text(entry)).casefold() and is_message(entry):
            hits.add(entry["history_id"])
            return True
        return False
    try:  # the strict rule decides every row read here; one it cannot place raises ValueError (``RoomResolver``)
        rows, before, gaps = reader.older(top, limit, counts, _SCAN_ROWS)
        importing = not chrono and 0 < reader.floor == before
        shown = [key for column, key in lens["chat"].keys if column == "appearance"]
        if shown and not chrono and before < top:
            gaps = set(gaps) | read_gaps(data_dir, thread_id, shown, before, top)
        has_more, beginning = before > 0 and not importing, before == 0 and not importing
        found = [entry for entry in reversed(rows) if entry["history_id"] in hits]
        # A legacy child's words are its card's too: its role kept in its task's metadata, or named by the page's
        # progress.
        named = _legacy_children(rooms, found) | _page_children(
            data_dir, rows, found, room_member_predicate(row_matches_thread, _stored_chat_id),
            beginning=beginning, recent=state is None)
    except HistoryCursorError:
        raise
    except ValueError:
        raise StoreError("Project room authority unavailable") from None  # as history's exact landing says it
    found = [entry for entry in found if not is_conversation_card_content(
        entry, str(entry.get("type") or ""), lambda row: str(row.get("task_id") or "") in named)]
    oldest = rows[0] if rows else None
    reached = (oldest.get("_chrono_key") or [str(oldest.get("ts") or "")])[0] if oldest else ""
    next_cursor = encode_cursor({"v": 1, "kind": "search", "chat_id": thread_id, "view": view, "q": digest,
                                 "upper": reader.upper, "before": before,
                                 "cut": reader.lower_key if chrono else None}) if has_more else None
    return {"query": query, "matches": [_match(entry, folded) for entry in found],
            "has_more": has_more, "next_cursor": next_cursor,
            "searched": {"rows": len(rows), "reached_ts": reached, "beginning": beginning,
                         "importing": importing, "gaps": sorted(gaps)}}


def _limit(value: Any) -> int:
    try:
        return max(1, min(int(value), _MAX_LIMIT))
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT


def make_chat_search_endpoint(data_dir: pathlib.Path):
    async def api_chat_search(request: Request) -> Response:
        """Search the room's saved messages; every read, the matching and the encode run off the event loop."""
        params = request.query_params
        try:
            thread_id = max(0, min(int(params.get("chat_id", 1)), 2**53 - 1)) or 1
        except (TypeError, ValueError):
            thread_id = 1
        query, cursor = params.get("q", ""), params.get("cursor") or None
        try:
            body = await asyncio.to_thread(lambda: json.dumps(
                search_room(data_dir, thread_id, query, cursor, _limit(params.get("limit", _DEFAULT_LIMIT))),
                ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (HistoryCursorError, OSError, StoreError) as exc:
            reason = (exc.reason if isinstance(exc, HistoryCursorError) else
                      "migrating" if isinstance(exc, StoreMigrating) else "history_source_unavailable")
            return Response(content=json.dumps({
                "query": _words(query)[:_MAX_QUERY_CHARS], "matches": [], "has_more": bool(cursor), "next_cursor": cursor,
                "searched": {"rows": 0, "reached_ts": "", "beginning": False, "importing": reason == "migrating", "gaps": []},
                "error": reason, "reason_code": reason}, ensure_ascii=False),
                status_code=exc.status if isinstance(exc, HistoryCursorError) else 503, media_type="application/json")
        return Response(content=body, media_type="application/json")

    return api_chat_search
