"""``GET /api/recall``: the neighbour door over the conversation, for every room.

A program on this computer (Claude Code, Codex, Cursor) finds the original rows of Ouroboros's conversation by
words and reopens one row, without waking Ouroboros: no model turn, no dispatch, no reply. Two exclusive forms:
``words`` (repeatable: alternatives) with ``room_id``, ``voice``, ``order``, ``date_from``, ``date_to``,
``count``, ``offset`` and ``snapshot``; or one ``row`` address. The search is the words reader's own
(``dialogue_recall.search``) over the rooms a person reads (Main, Projects, places: never the hidden partition
or synthetic traffic, by search or by address); this module only projects it outward.

Who may call is decided before this handler on the raw transport peer (``server_auth.recall_serving_app``); the
handler requires that admission again, so no other assembly of the app opens the door. Dialogue only: each row is
the closed ``RecallRow`` built here from its typed fields (``dialogue_words.fields_of``: a ``secret_request``
card is its non-value receipt), never a stored row serialized; no chronicle member, note, account, setting,
grant or attachment bytes; a row the reader does not admit (synthetic A2A traffic) is not found by address
either. Every printed reader is this route's or the ``ouroboros recall`` command's; the acting seal's address
and the mind's own tools are left out. Each admitted call appends one content-free ``dialogue_recall`` event
(mode, outcome) to ``logs/supervisor.jsonl``; a failed append never fails the read.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import urllib.parse
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from starlette.requests import Request
from starlette.responses import Response

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.server_auth import RECALL_CLOSED, RECALL_PEER

PAGE_DEFAULT, PAGE_MAX = 20, 100  # rows per page: the physical bound of one answer, said in the request's terms
CHAT_ID_LIMIT = 2**63  # a room address the store can hold: SQLite's signed 64-bit integer
# A field's speaker on the door, a closed set (``RecallField``): ``owner`` and ``host`` here, ``task <id>``,
# "the asking task" and "the closing task" from the words reader, else the row's own author label.
_FIELD_SPEAKERS = {"owner": "owner", "host": "host"}
_SEARCH_KEYS = ("words", "room_id", "voice", "order", "date_from", "date_to", "count", "offset", "snapshot")
_FILTER_KEYS = ("date_from", "date_to")
_HEADERS = {"Cache-Control": "no-store"}
_NOT_FOUND = "no dialogue row the door reads has this address"


UNSTABLE_READ = "The conversation changed during this read. Start again without snapshot or offset."
INVALID_SNAPSHOT = "Invalid snapshot. Use the snapshot from a search response or start again without it."


class Refusal(Exception):
    """One typed refusal: its status, code and sentence, and nothing of the store."""

    def __init__(self, status: int, code: str, error: str):
        super().__init__(error)
        self.status, self.code, self.error = status, code, error


def _reader(params: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """The same request as a path and as ``ouroboros recall`` arguments; a value that begins with a dash is
    written ``--name=value``, so the command line never reads it as an option."""
    cli = [part for key, text in params for part in (
        [f"--{key.replace('_', '-')}={text}"] if text.startswith("-") else [f"--{key.replace('_', '-')}", text])]
    return {"path": "/api/recall?" + urllib.parse.urlencode(list(params), quote_via=urllib.parse.quote),
            "cli": ["recall", *cli]}


def _outward(text: str) -> str:
    """A sentence of the words reader in the door's voice: its reader is another program, not Ouroboros."""
    return str(text).replace("my human's", "the owner's").replace("my human", "the owner")


def _speaker_label(row: Mapping[str, Any], author: Mapping[str, Any], api_door: bool) -> str:
    from ouroboros.dialogue_provenance import dialogue_provenance
    from ouroboros.dialogue_recall import API_DOOR_NOTE

    who = str(author.get("label") or "author not recorded")
    provenance = dialogue_provenance(row)
    if provenance and provenance not in who:
        who += f" [{provenance}]"
    return who + (f"; {API_DOOR_NOTE}" if api_door else "")


def row_view(hit: Mapping[str, Any], *, labels: Any, query: Any = None) -> Dict[str, Any]:
    """The closed outward projection of one hit in ``dialogue_recall.search``'s shape. Only the visible rooms that
    show the row are named (the caller admitted it by them). With ``query`` (a search answer) a plain message shows
    its matched field as an excerpt and a card every field; without it (a reopened row) every field whole."""
    from ouroboros.dialogue_recall import EXCERPT_CHARS
    from ouroboros.dialogue_words import excerpt

    row, written = hit["row"], int(hit["written_chat"])
    address = f"row:{written}@{row.get('ts') or ''}#{hit['sha256']}"  # the full digest: no short-prefix collision
    visible = [chat for chat in hit["rooms"] if chat > 0]
    own = written if written in visible else visible[0]
    fields = hit["fields"]
    card = len(fields) > 1 or bool(fields and fields[0]["name"] != "text")
    listed = fields if query is None or card else [field for field in fields if field["name"] in hit["matched"]]
    view: Dict[str, Any] = {
        "address": address, "room_id": str(own), "room": labels.label({"chat_id": own}),
        "rooms": [str(own), *(str(chat) for chat in visible if chat != own)],
        "ts": str(row.get("ts") or ""), "speaker": _speaker_label(row, hit["author"], hit["api_door"]),
        "matched": list(hit["matched"]),
        "fields": [{"name": field["name"], "speaker": _FIELD_SPEAKERS.get(field["speaker"]) or field["label"],
                    "text": field["text"] if query is None else excerpt(query, field["text"], EXCERPT_CHARS)}
                   for field in listed],
        "reopen": _reader([("row", address)])}
    if hit["reply_to"]:
        view["reply_to"] = hit["reply_to"]
        view["reply_unresolved"] = bool(hit["reply_unresolved"])
    return view


def _showing(root: pathlib.Path, row: Mapping[str, Any]) -> Tuple[List[int], str]:
    """The rooms that show ``row`` now, as the words reader resolves its hits (``dialogue_recall.Showing``: the
    store's own projection); with the registry unreadable, the chat it was written in, said."""
    from ouroboros.dialogue_recall import Showing

    showing = Showing(root, [row])
    return showing.rooms(row), showing.gap


def _integer(values: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    text = str(values.get(name, "") or "").strip()
    if not text:
        return default
    if not re.fullmatch(r"[0-9]{1,20}", text) or not low <= int(text) <= high:
        raise Refusal(400, "recall_bad_request", f"{name} is a whole number from {low} to {high}; got {text!r}")
    return int(text)


def search(root: pathlib.Path, words: List[str], values: Mapping[str, str]) -> Dict[str, Any]:
    """One ``RecallSearchResponse`` page from the words reader (``dialogue_recall.search``), visible rooms only."""
    from ouroboros import dialogue_recall
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.memory import Memory

    room_text, voice, order = (str(values.get(key, "") or "").strip() for key in ("room_id", "voice", "order"))
    if room_text and not _chat_id(room_text):  # the shared check's room sentence names the mind's own readers
        raise Refusal(400, "recall_bad_request", f"room_id is a room's chat id as text (1 is Main); got {room_text!r}")
    count = _integer(values, "count", PAGE_DEFAULT, 1, PAGE_MAX)
    offset = _integer(values, "offset", 0, 0, 2**53 - 1)
    filters = {key: values[key] for key in _FILTER_KEYS if str(values.get(key, "") or "").strip()}
    try:
        query, room = dialogue_recall.check_arguments(words, room_id=room_text, voice=voice, order=order,
                                                      filters=filters)
    except ValueError as exc:
        raise Refusal(400, "recall_bad_request", str(exc)) from None
    found = dialogue_recall.search(Memory(drive_root=root), query, room=room, voice=voice, order=order, count=count,
                                   offset=offset, snapshot=str(values.get("snapshot", "") or ""), filters=filters,
                                   visible_only=True)
    if found["refused_kind"] == "malformed":
        raise Refusal(400, "recall_bad_snapshot", INVALID_SNAPSHOT)
    if found["refused_kind"] == "unstable":
        raise Refusal(409, "recall_snapshot_changed", UNSTABLE_READ)
    if found["refused"]:
        raise Refusal(409, "recall_snapshot_changed", "the conversation or the query changed since this snapshot; "
                      "no mixed page was returned: start again without snapshot and offset")
    same = [("words", item) for item in words] + [(key, text) for key, text in (
        ("room_id", room_text), ("voice", voice), ("order", order), *((key, filters[key]) for key in filters))
        if text]
    current, hits, more = found["snapshot"], found["hits"], found["more"]
    following = None
    if current and more:  # the next page of this capture
        following = [*same, ("count", str(count)), ("offset", str(offset + len(hits))), ("snapshot", current)]
    notes = [_outward(note) for note in found["notes"]]
    if current is None:  # no continuation of a capture that moved or is still importing
        notes.append(UNSTABLE_READ)
    labels = RoomLabelResolver(root)
    return {"mode": "search", "words": list(words), "room_id": room_text, "voice": voice, "order": order or "newest",
            "total": found["total"], "observed": found["observed"], "offset": found["offset"], "more": more,
            "snapshot": current, "next": _reader(following) if following else None,
            "gaps": [_outward(gap) for gap in found["gaps"]], "notes": notes,
            "rows": [row_view(hit, labels=labels, query=query) for hit in hits]}


def reopen(root: pathlib.Path, text: str) -> Dict[str, Any]:
    """One ``RecallRowResponse``: the row at ``text`` whole, through the shared row resolver, when a visible room
    shows it; a hidden-only or synthetic row answers exactly as a missing one."""
    from ouroboros.chat_chain import parse_address, resolve_row, row_address, source_row_id
    from ouroboros.dialogue_provenance import RoomLabelResolver
    from ouroboros.dialogue_recall import field_speaker, label_authors
    from ouroboros.dialogue_words import fields_of

    try:
        wanted = parse_address(text)
    except ValueError:
        raise Refusal(400, "recall_bad_request", "row is an address as a search answer prints it: "
                                                 "row:<room>@<time>#<12 to 64 hex>") from None
    if not -CHAT_ID_LIMIT <= wanted["chat_id"] < CHAT_ID_LIMIT:
        raise Refusal(400, "recall_bad_request", "row names a room id the store cannot hold")
    if is_a2a_chat_id(wanted["chat_id"]):  # never resolved: an unadmitted row answers as a missing one
        raise Refusal(404, "recall_row_not_found", _NOT_FOUND)
    row, found = resolve_row(root, wanted)
    status = str(found.get("status") or "")
    if row is None and status == "row_ambiguous":
        raise Refusal(409, "recall_row_ambiguous", "several rows begin with these hex digits: give more of them")
    if row is None and status == "row_unreadable":
        raise Refusal(503, "recall_source_unreadable", "part of the conversation could not be read, so this "
                                                       "address can be neither found nor ruled out")
    if row is None or is_a2a_chat_id(row.get("chat_id")):
        raise Refusal(404, "recall_row_not_found", _NOT_FOUND)
    written = int(row_address(row)["chat_id"])
    rooms, rooms_note = _showing(root, row)
    if rooms_note:  # whether a person's room shows it cannot be decided: neither the row nor "not found"
        raise Refusal(503, "recall_source_unreadable", "part of the room rule could not be read, so this row can be "
                                                       "neither shown nor ruled out")
    if not any(chat > 0 for chat in rooms):  # the hidden partition is no room a person reads
        raise Refusal(404, "recall_row_not_found", _NOT_FOUND)
    authors, notes = label_authors(root, [(found["seq"], row)])  # as the search names it, lineage epoch included
    author = authors[found["seq"]]
    surface = row.get("client_surface") if isinstance(row.get("client_surface"), Mapping) else {}
    reply = row.get("reply_to") if isinstance(row.get("reply_to"), Mapping) else {}
    hit = {"row": row, "written_chat": written, "sha256": source_row_id(row), "rooms": rooms, "author": author,
           "api_door": surface.get("channel") == "api_command", "matched": [],
           "reply_to": str(reply.get("address") or ""), "reply_unresolved": bool(reply.get("unresolved")),
           "fields": [{"name": field.name, "text": field.text, "speaker": field.speaker,
                       "label": field_speaker(field, row, author)} for field in fields_of(row)]}
    return {"mode": "row", "row": row_view(hit, labels=RoomLabelResolver(root)),
            "notes": [_outward(item) for item in (*notes, rooms_note) if item]}


def _chat_id(text: str) -> bool:
    """A room address as text: an integer the store can hold."""
    return bool(re.fullmatch(r"-?[0-9]{1,19}", text)) and -CHAT_ID_LIMIT <= int(text) < CHAT_ID_LIMIT


def answer(root: pathlib.Path, items: Sequence[Tuple[str, str]]) -> Tuple[int, Dict[str, Any], str]:
    """``(status, body, mode)`` for one admitted request's query items; ``mode`` is search, row or invalid."""
    from ouroboros.store import StoreError, StoreMigrating

    keys = [key for key, _value in items]
    mode = "row" if "row" in keys and "words" not in keys else "search" if "words" in keys and "row" not in keys \
        else "invalid"
    try:
        unknown = sorted(set(keys) - {*_SEARCH_KEYS, "row"})
        repeated = sorted({key for key in keys if key != "words" and keys.count(key) > 1})
        if unknown or repeated or mode == "invalid" or (mode == "row" and keys != ["row"]):
            raise Refusal(400, "recall_bad_request", "give words (one or more, each an alternative) with optional "
                          f"{', '.join(_SEARCH_KEYS[1:])}, or one row address alone"
                          + (f"; unknown: {', '.join(unknown)}" if unknown else "")
                          + (f"; given twice: {', '.join(repeated)}" if repeated else ""))
        values = dict(items)
        body = (reopen(root, values["row"]) if mode == "row"
                else search(root, [value for key, value in items if key == "words"], values))
        return 200, body, mode
    except Refusal as refusal:
        return refusal.status, {"error": refusal.error, "code": refusal.code}, mode
    except StoreMigrating:
        return 503, {"error": "the conversation is still importing; ask again shortly", "code": "migrating"}, mode
    except (StoreError, OSError):
        return 503, {"error": "the conversation store could not be read", "code": "recall_source_unavailable"}, mode


def _count(root: pathlib.Path, mode: str, status: int, body: Mapping[str, Any]) -> None:
    """One content-free event per admitted call: no words, address, room or sender, only mode and outcome."""
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(root / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "dialogue_recall", "mode": mode,
            "outcome": "ok" if status == 200 else str(body.get("code") or status)})
    except Exception:
        pass


def make_recall_endpoint(data_dir: pathlib.Path):
    async def api_recall(request: Request) -> Response:
        """Search or reopen admitted dialogue rows for a program on this computer; reads only, off the event loop."""
        if request.scope.get(RECALL_PEER) is not True:  # admitted by the raw peer, or closed like the gate closes
            return Response(json.dumps(RECALL_CLOSED), status_code=403, media_type="application/json",
                            headers=_HEADERS)
        root = pathlib.Path(data_dir)

        def run() -> Tuple[int, bytes]:
            status, body, mode = answer(root, request.query_params.multi_items())
            _count(root, mode, status, body)
            return status, json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        status, content = await asyncio.to_thread(run)
        return Response(content, status_code=status, media_type="application/json", headers=_HEADERS)

    return api_recall
