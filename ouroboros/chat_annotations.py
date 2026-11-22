"""Routing receipt journal. A physical source address identifies the row on screen,
and a routing token identifies each act. The picker claims its card under the
same append lock as the writer and compactor.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from typing import Any, Dict, Optional

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import iter_jsonl_objects, jsonl_append_lock_path, replace_atomic, utc_now_iso

log = logging.getLogger(__name__)

_ANNOTATIONS_NAME = "chat_annotations.jsonl"
# Receipt id for a routing act that belongs to NO owner message — the agent's
# own steer of a task it already routed for, or of one it was told about after
# its origin. The act still needs a durable token-bound receipt (the tool waits
# on one through ``routing_wait``, and silence there reports a landed delivery
# as unconfirmed), but it must annotate no owner message: the id is the act's
# routing token, so nothing in any chat joins to it. Chat-row membership is
# therefore the wrong retention test for these rows — compaction applies the
# chat-retention rule only to ids that address a real message, and bounds these
# by the newest-N cap below. Per-id dedupe cannot bound them: every steer mints
# a fresh token, so each act has an id of its own. The colon shape cannot
# collide with a client id: the browser mints ``msg-<epoch_ms>-<n>`` and non-web
# ingress ``host-<uuid5>``.
AGENT_RECEIPT_ID_PREFIX = "agent-steer:"
# How many synthetic receipts survive a compaction. ``routing_wait`` polls for at
# most 15 seconds, so every live waiter's row is far inside this window; without a
# cap a long-lived install would keep the file permanently above the threshold and
# rewrite the whole of it on every append.
_RETAINED_AGENT_RECEIPTS = 64
_COMPACT_AT_BYTES = 800_000


def _annotation_index(path: pathlib.Path, *, by_token=False) -> dict:
    latest = {}
    for row in iter_jsonl_objects(path):
        message_id = str(row.get("client_message_id") or "")
        room = row.get("source_chat_id")
        if message_id and row.get("type") == "chat_annotation" and (room is None or type(room) is int):
            key = (room, message_id)
            if by_token:
                key += (str(row.get("routing_token") or ""),)
            address = row.get("source_row_address", "")
            if not isinstance(address, str):
                continue
            # ``source_row_address: ""`` is the host's stated "unplaced" (``explicit_source_fields``); an OMITTED key is
            # an older receipt. The two never share a key, so neither erases the other's act under one correlation key.
            key += (None if "source_row_address" in row and not address else address,)
            latest.pop(key, None)
            latest[key] = row
    return latest


def _resolve_annotation_rooms(drive_root, index):
    from ouroboros.dialogue_store import annotation_rooms
    from ouroboros.store import StoreError
    try:
        legacy = annotation_rooms(drive_root, (key[1] for key in index if key[0] is None))
    except (OSError, StoreError):
        legacy = {}  # explicit room receipts remain readable while old rows cannot be placed
    resolved = {}
    for (room, message, *token), row in index.items():
        key = (legacy.get(message) if room is None else room, message, *token)
        resolved.pop(key, None)
        resolved[key] = row
    return resolved


def scoped_chat_annotations(drive_root: Any, *, by_token=False) -> dict:
    """Latest annotation for each source room and message. Old unscoped rows need one identifiable source."""
    return _compatible_annotations(pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME, by_token=by_token, scoped=True)


def receipt_source_room(ref, message, fallback):
    """A bound ingress row names a receipt's room even when the turn executes elsewhere."""
    from ouroboros.project_dialogue import owner_message_ref_is_valid

    return (ref["chat_id"] if message and owner_message_ref_is_valid(ref) and ref.get("row_sha256")
            and ref["client_message_id"] == message else fallback)


def routing_source_address(ref, room, message):
    """The physical source an act identifies, when its room and client correlation agree."""
    from ouroboros.chat_chain import format_address

    source_room = receipt_source_room(ref, message, None)
    return format_address(ref) if source_room is not None and source_room == room else ""


def explicit_source_fields(root, ref, room, message, fields, *, suppressed=False, selected=False, picker=False):
    """The placement a receipt states for an act that named an EXPLICIT owner source, else ``{}``.

    Explicit: a provided reference of any shape (a weak or malformed one included), the door's suppression of the origin
    (``suppressed``), or a source the act selected (``selected``). A positive resolution is reused: the row identity the
    host minted (``routing_source_address``), else the one canonical row an older reference names. What the host cannot
    verify is saved as ``source_row_address: ""``: the key's presence with an empty value is the stated outcome "no row
    wears this", so neither the live frame nor a reload places it by client id, and it is never read as an older receipt
    (an OMITTED key: one authoritative legacy match may place it later). An act that named no source, a picker (its
    addressless authority keeps its decisions) and a receipt already placed return ``{}``."""
    if picker or fields.get("source_row_address") or not message:
        return {}
    named = isinstance(ref, dict) and bool(ref) and not suppressed
    if not (named or suppressed or selected):
        return {}
    if named:
        if address := routing_source_address(ref, room, message):
            return {"source_row_address": address}
        from ouroboros.chat_chain import format_address, row_address
        from ouroboros.project_dialogue import owner_message_ref_is_valid, resolve_owner_message_source

        if (owner_message_ref_is_valid(ref) and not ref.get("row_sha256")
                and (ref["chat_id"], ref["client_message_id"]) == (room, message)):
            try:
                if row := resolve_owner_message_source(root, ref):
                    return {"source_row_address": format_address(row_address(row))}
            except Exception:  # an unreadable store verifies nothing: the stated outcome is unplaced, the act still lands
                log.debug("Explicit routing source not verifiable", exc_info=True)
    return {"source_row_address": ""}


def _placed_annotations(root, *, by_token=False, message_ids=None, weak_readers=()):
    """Resolve all older id-only receipts and readers in one indexed lookup, preserving append order.
    A row that states an EMPTY ``source_row_address`` was decided unplaced when written: it is never an older receipt."""
    from ouroboros.dialogue_store import annotation_sources
    from ouroboros.store import StoreError

    index = _annotation_index(pathlib.Path(root) / "logs" / _ANNOTATIONS_NAME, by_token=by_token)
    if message_ids is not None:
        index = {key: row for key, row in index.items() if key[1] in message_ids}
    weak = {key[1] for key in index if key[-1] == ""}  # omitted keys only: a stated-unplaced key is ``None``
    weak.update(weak_readers)
    try:
        legacy = annotation_sources(root, weak) if index else {}
    except (OSError, StoreError):
        legacy = {}
    placed = {}
    for (room, message, *token, address), row in index.items():
        if address is None:
            address = ""  # stated unplaced: it stays in its own room, wearing no row
        elif not address:
            room, address = legacy.get((room, message)) or (room, "")
            if address and row.get("action") == "steer_task" and row.get("status") == "read":
                row = {**row, "source_row_address": address}
        key = (room, message, address, *token)
        placed.pop(key, None)
        placed[key] = row
    return placed, legacy


def message_receipts(root, sources):
    """Read each held message's acts. A room/id alone can read an exact act only for one identifiable row; an act
    that wears no address is not read by message/id alone (its issuing token still reads it)."""
    messages = [(room, str(message), extra[0] if extra else "") for room, message, *extra in sources if message]
    if not messages:
        return []
    placed, legacy = _placed_annotations(root, by_token=True, message_ids={msg for _, msg, _ in messages},
                                         weak_readers=[msg for _, msg, address in messages if not address])
    requested = set()
    for room, msg, address in messages:
        identity = (room, address) if address else legacy.get((room, msg), (room, ""))
        if identity and identity[0] is not None and identity[1]:
            requested.add((identity[0], msg, identity[1]))
    return [(key[1], row) for key, row in placed.items() if key[:3] in requested]


def addressed_chat_annotations(root, *, by_token=False, message_ids=None):
    """Only receipts a physical owner row can wear."""
    placed, _ = _placed_annotations(root, by_token=by_token, message_ids=message_ids)
    return {key: row for key, row in placed.items() if key[2]}


def placed_token_receipts(root, message_ids, token):
    """One act's saved facts per source, including a legacy fact that remains unplaced."""
    placed, _ = _placed_annotations(root, by_token=True, message_ids=message_ids)
    return {key[:3]: row for key, row in placed.items() if key[3] == token}


def latest_picker_annotation(root, message_id, token):
    """The latest act on this picker's original, or the one known status of a superseded token."""
    placed, _ = _placed_annotations(root, by_token=True, message_ids={message_id})
    latest, owners = {}, set()
    for (room, _, address, act), row in placed.items():
        latest[room, address] = row
        if act == token:
            owners.add((room, address))
    if len(owners) == 1:
        source = owners.pop()
        return source[0], latest[source]
    return None, next(iter(latest.values())) if len(latest) == 1 else {}


def _compatible_annotations(path, *, by_token=False, scoped=False):
    index = _annotation_index(path, by_token=by_token)
    if scoped or len(index) != len({key[1:-1] for key in index}):
        index = _resolve_annotation_rooms(path.parent.parent, index)
    latest, rooms, addresses = {}, {}, {}
    for (room, message, *token, address), row in index.items():
        if scoped and room is None:
            continue
        key = (room, message, *token) if scoped else ((message, *token) if by_token else message)
        # A token-bound act can have an older roomless receipt from the same act.
        if room is not None or not by_token:
            rooms.setdefault(key, set()).add(room)
        if address:
            addresses.setdefault(key, set()).add(address)
        latest.pop(key, None)
        latest[key] = row
    return {key: row for key, row in latest.items()
            if len(rooms.get(key, ())) <= 1 and len(addresses.get(key, ())) <= 1}


def latest_chat_annotations(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Latest presentation annotation per message; a torn tail is ignored."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    return _compatible_annotations(path)


def chat_annotation_receipt(drive_root: Any, client_message_id: str, routing_token: str, *,
                            source_chat_id: Optional[int] = None, source_row_address: Optional[str] = None) -> Dict[str, Any]:
    """The latest annotation written for one routing attempt, read BY TOKEN.

    A later act under the same owner message no longer hides an earlier act's
    receipt: the waiter that minted the token polls for its own outcome, and the
    UI projection stays latest-per-message (``latest_chat_annotations``).
    ``source_chat_id``: the reader knows the message's room too (one act quoting
    two rooms' messages), so it reads that room's row and no other."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    message, token = str(client_message_id or ""), str(routing_token or "")
    if source_row_address is not None:
        return dict(placed_token_receipts(drive_root, {message}, token).get((source_chat_id, message, source_row_address)) or {})
    row = (scoped_chat_annotations(drive_root, by_token=True).get((int(source_chat_id), message, token))
           if source_chat_id is not None else _compatible_annotations(path, by_token=True).get((message, token)))
    return dict(row) if row else {}


def _compact_annotations_locked(drive_root: Any, path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size < _COMPACT_AT_BYTES:
        return
    from ouroboros import dialogue_store as ds

    # Incomplete history cannot prove an annotation's source is absent. Leave the
    # sidecar intact until its retained-message decisions can be made truthfully;
    # over an unreadable line (a gap) no source is proven absent, but the newest
    # act per message still replaces the older ones.
    coverage = ds.range_coverage(drive_root, "chat")
    if not coverage["imported"]:
        return
    index = _annotation_index(path, by_token=True)
    latest = {key[1]: row for key, row in index.items()}
    kept_receipts = set(sorted(
        (message_id for message_id in latest if message_id.startswith(AGENT_RECEIPT_ID_PREFIX)),
        key=lambda message_id: str(latest[message_id].get("ts") or ""),
    )[-_RETAINED_AGENT_RECEIPTS:])
    named = [message_id for message_id in latest if not message_id.startswith(AGENT_RECEIPT_ID_PREFIX)]
    retained_ids = set(named) if coverage["gaps"] else ds.carried_ids(drive_root, named)
    # Retained per (message, token): an older act's receipt under a message that
    # is still in the chat survives beside the message's newest act.
    rows = [
        row for (_room, message_id, _token, _address), row in index.items()
        if message_id in retained_ids or message_id in kept_receipts
    ]
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    replace_atomic(tmp, path)


# One owner-routing receipt's fields (#968): its row below, its live frame (`send_routing_ack`) and its history
# replay state this one set; the destination part names the room and the works its words reached.
ROUTING_DESTINATION_FIELDS = ("project_id", "project_chat_id", "project_name", "target_task_ids", "target_titles")
ROUTING_RECEIPT_FIELDS = ("source_row_address", "action", "target", "target_label", "status", "routing_token", "cause", "options",
                          "attachment_manifest", "reasoning_effort", *ROUTING_DESTINATION_FIELDS)


def append_chat_annotation(
    drive_root: Any,
    client_message_id: str,
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    project_id: str = "",
    project_chat_id: int = 0,
    status: str,
    routing_token: str = "",
    reason: str = "",
    detail: str = "",
    cause: str = "",
    options: Any = None,
    attachment_manifest: Any = None,
    require_latest_status: Any = None,
    require_latest_token: Any = None,
    source_chat_id: Optional[int] = None,
    reasoning_effort: str = "", source_row_address: str = "",
    project_name: str = "", target_task_ids: Any = None, target_titles: Any = None,
    source_unplaced: bool = False,
) -> bool:
    """Append one compact UI annotation.

    Presentation-first with ONE named exception (#198): a routing refusal row
    (status=needs_manual_target) is also the picker's durable decision-card
    authority — its token+options validate the owner's click, and the
    dispatch_pending/closing rows carry the click's first-wins/idempotency
    facts. Routing STATE still lives in the supervisor receipts (task-result
    admission, mailbox); the sidecar only arbitrates the card.

    ``source_unplaced`` writes ``source_row_address: ""`` (``explicit_source_fields``) instead of leaving the key
    out: an empty ``source_row_address`` argument alone still means "no address stated" (an older receipt).

    ``require_latest_status`` (a set of status strings) turns the append into
    a compare-and-append under the annotations lock: the row is written only
    while the message's CURRENT latest status is in the set — the first-wins
    claim seam of the routing picker (#198). Absent/None keeps plain append.

    """
    message_id = str(client_message_id or "").strip()
    if not message_id:
        return False
    row = {
        "ts": utc_now_iso(),
        "type": "chat_annotation",
        "client_message_id": message_id[:200],
        **({"source_chat_id": source_chat_id} if source_chat_id is not None else {}),
        **({"source_row_address": source_row_address} if source_row_address
           else {"source_row_address": ""} if source_unplaced else {}),
        "action": str(action or "")[:80],
        "target": str(target or "")[:200],
        "status": str(status or "")[:80],
    }
    if target_task_ids is not None:
        row.update(project_name=project_name, target_task_ids=list(target_task_ids), target_titles=list(target_titles or []))
    if str(target_label or ""):
        row["target_label"] = str(target_label)[:200]
    if project_id and project_chat_id:
        row.update(project_id=str(project_id), project_chat_id=int(project_chat_id))
    if str(routing_token or ""):
        row["routing_token"] = str(routing_token)[:128]
    if str(reason or ""):
        row["reason"] = str(reason)[:200]
    if str(detail or ""):
        row["detail"] = str(detail)[:1000]
    if str(cause or ""):
        # Q3=A: the host-owned owner-facing sentence for a refused act; the
        # browser renders it verbatim on the receipt line (reason stays a code).
        row["cause"] = str(cause)[:200]
    if isinstance(options, list):
        row["options"] = [dict(item) for item in options[:100] if isinstance(item, dict)]
    if isinstance(attachment_manifest, list):
        row["attachment_manifest"] = [
            dict(item) for item in attachment_manifest if isinstance(item, dict)
        ]
    if str(reasoning_effort or ""):
        row["reasoning_effort"] = str(reasoning_effort)[:16]
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0, owner_aware_stale=True)
    if lock_fd is None:
        return False
    try:
        if require_latest_status is not None or require_latest_token is not None:
            if source_row_address:
                latest = addressed_chat_annotations(drive_root, message_ids={message_id}).get(
                    (source_chat_id, message_id, source_row_address))
            else:
                latest = (scoped_chat_annotations(drive_root).get((source_chat_id, message_id))
                          if source_chat_id is not None else latest_chat_annotations(drive_root).get(message_id))
            if latest is None:
                return False
            if require_latest_status is not None and latest.get("status", "") not in require_latest_status:
                return False
            if require_latest_token is not None and latest.get("routing_token", "") not in require_latest_token:
                return False
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        _compact_annotations_locked(drive_root, path)
        return True
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
