"""Messages I posted through a tool acting as an account, kept by provider identity so they never read as my human's.

A tool may post as an account a connector also reads: an MCP server signed in as my human posts into a channel the
connector hears, and that post comes back as an arrival from my human's confirmed account. A message I sent is never
my human's words, whatever its actor (BIBLE P1, provenance). ``record_result`` keeps the provider identity
(conversation id, message id) that a successful MCP call's result names as the one message it posted, when the
server declares the tool a write (``readOnlyHint: false``). A tool without that declaration keeps nothing: a read
naming one message (a thread holding only my human's post) must never turn their words into mine, so an undeclared
send stays a disclosed gap. ``places.place_owner_accounts`` carries the kept identities into every owner
decision (``places.owner_author``), which compares identities only, never words: my human repeating my words in their
own message is still my human.

A result names its message by the provider's own fields: a JSON object, or a CSV row under its header, holding both a
conversation field (``channel``, ``channel_id``, ``conversation_id``, ``chat_id``) and a message field (``ts``,
``message_ts``, ``message_id``, ``msgid``), compared without case or punctuation. A result naming no such pair, or
several (a listing, not a receipt), keeps nothing: that send stays a disclosed gap, never a guess from its words.
The record lands when the call's result returns; an echo the connector delivers before that is read again at every
later use (a turn's opening, its drain, a Safety check, a history view), and only a decision taken in that moment
(an admission's snapshot) can still have seen it as my human's.
Connector sends are not kept here: a connector never returns its own sends as arrivals (CREATING_SKILLS "Who the
connector writes as").

``state/own_sends.json`` holds the newest ``KEEP`` identities (about 120 bytes each); every write drops the oldest. An
unreadable document reads as no recorded send and the next record starts it afresh: it only keeps my human's name off
my echoes, and an echo it no longer holds reads by its actor, as before this record existed.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from ouroboros.utils import update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

FILENAME = "own_sends.json"
KEEP = 500
_CONVERSATION = frozenset({"channel", "channelid", "conversationid", "chatid"})
_MESSAGE = frozenset({"ts", "messagets", "messageid", "msgid"})


def _field(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def own_sends_path(drive_root: Any) -> Path:
    return Path(drive_root) / "state" / FILENAME


def message_identity(conversation_id: Any, message: Any) -> tuple[str, str] | None:
    """A place message's provider identity: its conversation id and the connector's ``message.message_id`` (the live
    event contract, CREATING_SKILLS); None when either is missing."""
    found = str((message if isinstance(message, Mapping) else {}).get("message_id") or "").strip()
    conversation = str(conversation_id or "").strip()
    return (conversation, found) if conversation and found else None


def _object_pairs(value: Any, pairs: set) -> None:
    if isinstance(value, list):
        for item in value:
            _object_pairs(item, pairs)
    elif isinstance(value, Mapping):
        named = {_field(key): item for key, item in value.items() if isinstance(item, (str, int)) and str(item).strip()}
        found = [next((str(named[key]) for key in named if key in group), "") for group in (_CONVERSATION, _MESSAGE)]
        if all(found):
            pairs.add((found[0].strip(), found[1].strip()))
        for item in value.values():
            _object_pairs(item, pairs)


def named_messages(text: Any) -> set[tuple[str, str]]:
    """Every (conversation id, message id) a tool result names: its JSON (whole, or one of its parts, a
    ``structuredContent`` part included), else its CSV rows under the first header naming both fields."""
    body, pairs = str(text or ""), set()
    for part in (body, *body.split("\n\n")):
        part = part.strip().removeprefix("structuredContent:").strip()
        if part[:1] in ("{", "["):
            try:
                _object_pairs(json.loads(part), pairs)
            except ValueError:
                continue
    if pairs:
        return pairs
    columns = None
    for row in csv.reader(io.StringIO(body)):
        if columns is None:
            keys = [_field(cell) for cell in row]
            found = [next((i for i, key in enumerate(keys) if key in group), None) for group in (_CONVERSATION, _MESSAGE)]
            columns = found if None not in found else None
        elif len(row) > max(columns) and row[columns[0]].strip() and row[columns[1]].strip():
            pairs.add((row[columns[0]].strip(), row[columns[1]].strip()))
    return pairs


def record_result(drive_root: Any, metadata: Any, text: Any, *, via: str) -> bool:
    """Keep the one message a successful MCP call's result names as posted, for a tool its server declares a write
    (``readOnlyHint: false``); any other tool, a result naming no message or several, and a store that cannot be
    written keep nothing (logged). True when kept."""
    annotations = (metadata or {}).get("annotations") if isinstance(metadata, Mapping) else None
    if drive_root is None or not isinstance(annotations, Mapping) or annotations.get("readOnlyHint") is not False:
        return False
    try:
        pairs = named_messages(text)
    except (csv.Error, RecursionError, ValueError) as exc:  # an unreadable result names nothing; the call stands
        log.warning("Own send %s: result unreadable (%s)", via, exc)
        return False
    if len(pairs) != 1:
        return False
    (conversation, message), = pairs
    row = {"conversation_id": conversation, "message_id": message, "at": utc_now_iso(), "via": via}

    def keep(current: dict) -> dict:
        rows = current.get("sent") if isinstance(current.get("sent"), list) else []
        return {"schema_version": 1, "sent": [*(item for item in rows if isinstance(item, dict)), row][-KEEP:]}

    try:
        update_json_locked(own_sends_path(drive_root), keep)
    except (OSError, ValueError, TimeoutError) as exc:
        log.warning("Own send %s could not be kept: %s", via, exc)
        return False
    return True


def sent_identities(drive_root: Any) -> frozenset:
    """The kept (conversation id, message id) identities, read once; none without a drive or a readable store."""
    path = own_sends_path(drive_root) if drive_root is not None else None
    if path is None or not path.exists():
        return frozenset()
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("sent") or []
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("Own sends unreadable at %s: %s", path, exc)
        return frozenset()
    return frozenset((str(row["conversation_id"]), str(row["message_id"])) for row in rows
                     if isinstance(row, dict) and row.get("conversation_id") and row.get("message_id"))
