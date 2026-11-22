"""``GET /api/recall``: the neighbour door over the conversation, for every room.

A program on this computer (Claude Code, Codex, Cursor) finds the original rows of Ouroboros's conversation by
words and reopens one row, without waking Ouroboros: no model turn, no dispatch, no reply. Two exclusive forms:
``words`` (repeatable: alternatives) with ``room_id``, ``voice``, ``order``, ``date_from``, ``date_to``,
``count``, ``offset`` and ``snapshot``; or one ``row`` address. The search is the words reader's own
(``dialogue_recall.collect``) over every room it admits; this module only projects it outward.

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from starlette.requests import Request
from starlette.responses import Response

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.server_auth import RECALL_CLOSED, RECALL_PEER

PAGE_DEFAULT, PAGE_MAX = 20, 100  # rows per page: the physical bound of one answer, said in the request's terms
CHAT_ID_LIMIT = 2**63  # a room address the store can hold: SQLite's signed 64-bit integer
# A field's speaker on the door, a closed set (``RecallField``): ``owner`` and ``host`` here, ``task <id>``,
# "the asking task" and "the closing task" from the words reader, else the row's own author label.
_FIELD_SPEAKERS = {"owner": "owner", "host": "host"}
OWNER_ACCOUNTS_NOTE = "owner accounts unreadable: no place row is labelled as the owner's"
_SEARCH_KEYS = ("words", "room_id", "voice", "order", "date_from", "date_to", "count", "offset", "snapshot")
_FILTER_KEYS = ("date_from", "date_to")
_HEADERS = {"Cache-Control": "no-store"}
_NOT_FOUND = "no dialogue row the door reads has this address"


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


def _full_address(row: Mapping[str, Any]) -> str:
    from ouroboros.chat_chain import row_address

    address = row_address(row)  # a SourceProjection keeps its source body's identity
    return f"row:{address['chat_id']}@{address['ts']}#{address['row_sha256']}"


def _speaker_label(row: Mapping[str, Any], author: Mapping[str, Any]) -> str:
    from ouroboros.dialogue_provenance import dialogue_provenance
    from ouroboros.dialogue_recall import API_DOOR_NOTE

    who = str(author.get("label") or "author not recorded")
    provenance = dialogue_provenance(row)
    if provenance and provenance not in who:
        who += f" [{provenance}]"
    surface = row.get("client_surface") if isinstance(row.get("client_surface"), Mapping) else {}
    return who + (f"; {API_DOOR_NOTE}" if surface.get("channel") == "api_command" else "")


def row_view(root: pathlib.Path, row: Mapping[str, Any], *, author: Mapping[str, Any], fields: Sequence[Any],
             matched: Sequence[str], rooms: Any, shown_in: Optional[int] = None, query: Any = None) -> Dict[str, Any]:
    """The closed outward projection of one row. With ``query`` (a search answer) a plain message shows its matched
    field as an excerpt and a card every field; without it (a reopened row) every field whole."""
    from ouroboros.dialogue_recall import _excerpt, _speaker  # shared names once the words reader publishes them
    from ouroboros.room_appearances import frame_rooms

    address = _full_address(row)
    own = address[len("row:"):].split("@", 1)[0]
    shown = [own]
    for room in (shown_in, *(frame_rooms(root, row, stamped_only=True).get("also_rooms") or [])):
        if room is not None and str(room) not in shown:
            shown.append(str(room))
    card = len(fields) > 1 or bool(fields and fields[0].name != "text")
    listed = fields if query is None or card else [field for field in fields if field.name in matched]
    view: Dict[str, Any] = {
        "address": address, "room_id": own, "room": rooms.label(row), "rooms": shown,
        "ts": str(row.get("ts") or ""), "speaker": _speaker_label(row, author), "matched": list(matched),
        "fields": [{"name": field.name, "speaker": _FIELD_SPEAKERS.get(field.speaker) or _speaker(field, row, author),
                    "text": field.text if query is None else _excerpt(field.text, query)} for field in listed],
        "reopen": _reader([("row", address)])}
    reply = row.get("reply_to") if isinstance(row.get("reply_to"), Mapping) else {}
    if str(reply.get("address") or ""):
        view["reply_to"] = str(reply["address"])
        view["reply_unresolved"] = bool(reply.get("unresolved"))
    return view


def _owner_accounts(root: pathlib.Path) -> Tuple[Optional[Mapping[str, Any]], str]:
    """``(accounts, note)``: the places' confirmed owner accounts, or None and the note the answer carries."""
    try:
        from ouroboros.places import place_owner_accounts

        return place_owner_accounts(root), ""
    except Exception as exc:  # no place row is then labelled as the owner's, and the answer says so
        return None, f"{OWNER_ACCOUNTS_NOTE} ({type(exc).__name__})"


def _integer(values: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    text = str(values.get(name, "") or "").strip()
    if not text:
        return default
    if not re.fullmatch(r"[0-9]{1,20}", text) or not low <= int(text) <= high:
        raise Refusal(400, "recall_bad_request", f"{name} is a whole number from {low} to {high}; got {text!r}")
    return int(text)


def search(root: pathlib.Path, words: List[str], values: Mapping[str, str]) -> Dict[str, Any]:
    """One ``RecallSearchResponse`` page from the words reader (``dialogue_recall.collect``)."""
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
    found = dialogue_recall.collect(Memory(drive_root=root), query, room=room, voice=voice, order=order,
                                    filters=filters)
    rows, coverage, current = found["rows"], found["coverage"], found["snapshot"]
    requested = str(values.get("snapshot", "") or "").strip().lower()
    if requested and (not coverage.get("snapshot_stable") or requested != current):
        raise Refusal(409, "recall_snapshot_changed", "the conversation or the query changed since this snapshot; "
                      "no mixed page was returned: start again without snapshot and offset")
    page = rows[offset:offset + count]
    more = max(0, len(rows) - offset - len(page))
    stable = bool(coverage.get("snapshot_stable"))
    same = [("words", item) for item in words] + [(key, text) for key, text in (
        ("room_id", room_text), ("voice", voice), ("order", order), *((key, filters[key]) for key in filters))
        if text]
    following = [*same, ("count", str(count)), ("offset", str(offset + len(page)))] + (
        [("snapshot", current)] if stable else [])
    labels = RoomLabelResolver(root)
    # the words reader's note in its own voice becomes the door's; under voice=owner it is already a gap
    notes = [OWNER_ACCOUNTS_NOTE] if found.get("said") and voice != "owner" else []
    if more and not stable:
        notes.append("no continuation: the capture did not stabilize (part of the conversation is still importing); "
                     "ask again from offset 0 when the import has finished")
    return {"mode": "search", "words": list(words), "room_id": room_text, "voice": voice, "order": order or "newest",
            "total": None if found["gaps"] else len(rows), "observed": len(rows), "offset": offset, "more": more,
            "snapshot": current if stable else None, "next": _reader(following) if more and stable else None,
            "gaps": list(found["gaps"]), "notes": notes,
            "rows": [row_view(root, row, **found["hits"][id(row)], rooms=labels, shown_in=room, query=query)
                     for row in page]}


def reopen(root: pathlib.Path, text: str) -> Dict[str, Any]:
    """One ``RecallRowResponse``: the admitted row at ``text`` whole, through the shared row resolver."""
    from ouroboros.chat_chain import parse_address, resolve_row
    from ouroboros.dialogue_provenance import RoomLabelResolver, row_author
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
    accounts, note = _owner_accounts(root)
    author = row_author(row, owner_accounts=accounts)
    return {"mode": "row", "row": row_view(root, row, author=author, fields=fields_of(row), matched=[],
                                           rooms=RoomLabelResolver(root)), "notes": [note] if note else []}


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
