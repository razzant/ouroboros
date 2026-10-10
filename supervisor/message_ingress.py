"""Named ingress acceptance: one canonical inbound row per message id, landed before any dispatch.

The owner of a named owner/skill message's acceptance (ARCHITECTURE 04 "WebSocket chat", 12
"Operation correlation"): a same-id redelivery rejoins the accepted row when its words and ordered
attachment content match and is refused otherwise, ``_INGRESS_LOCK`` serializes check → canonical
row → dispatch, and only THIS process knows what happened to a row's dispatch: ``_UNDISPATCHED``
(its write raised before dispatch, so one retry hands it over) and the row's ``ingress_process``
stamp (which process accepted it — ``accepted_here``). ``_DISPATCH_ENTERED`` records actual
call entry separately; acceptance cannot prove it, even in this process. These facts do not
survive the process, and nothing durable records dispatch (``delivery_facts`` states them).
Which row an id already names comes from ``_AcceptedIds``, this process's index over the
retained chat chain. ``supervisor.message_bus`` re-exports the public names; its ``DATA_DIR``
and ``log_chat`` are read at call time.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

from ouroboros.chat_uploads import attachment_views, same_message, stored_refs
from ouroboros.utils import JsonlChainUnreadable, jsonl_chain_handles, utc_now_iso

log = logging.getLogger(__name__)

_INGRESS_LOCK = threading.Lock()
# Named acceptances (web or skill) whose canonical write raised in THIS process: the row may have
# landed and dispatch was never entered — positive proof, which no row from before a restart has.
# Operation reads say ``lost`` and history marks the row ``ingress_undispatched``; the first explicit
# same-id retry of the same message takes the proof and dispatches the row once, and a later landed
# write of the id clears it. The process's end retires the rest: then nothing is known, nothing replays.
_UNDISPATCHED: "set[Tuple[int, str]]" = set()
# Positive call-entry evidence, separate from acceptance and never persisted.
_DISPATCH_ENTERED: set[tuple] = set()


def _dispatch_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in ("ingress_process", "chat_id", "client_message_id", "ts"))


def dispatch_entered(row: dict) -> bool:
    return accepted_here(row) and _dispatch_key(row) in _DISPATCH_ENTERED


def _enter_dispatch(row: dict) -> None:
    _DISPATCH_ENTERED.add(_dispatch_key(row))


def acceptance_undispatched(chat_id: int, client_message_id: str) -> bool:
    return (int(chat_id), str(client_message_id)) in _UNDISPATCHED


def accepted_here(row: dict) -> bool:
    """True when THIS host process wrote that named acceptance row (``log_chat`` stamps the process
    generation as ``ingress_process``). This proves acceptance only, never dispatch entry;
    a row from an ended process (a restart came between) or before the stamp is unknown."""
    from ouroboros.process_custody import current_custody_session_id

    stamp = str((row or {}).get("ingress_process") or "")
    return bool(stamp) and stamp == current_custody_session_id()


def delivery_facts(row: dict, chat_id: int) -> dict:
    """What this process knows of an accepted row's dispatch, as its echo and history row say it.

    ``ingress_undispatched``: its write raised before dispatch (one retry hands it over);
    ``ingress_dispatched``: dispatch was entered here; ``ingress_pending``: this live process took
    it and has entered or refused neither yet (an append still returning, an echo sent before its
    deferred dispatch) — a later echo or read says which. Nothing: a row of an ended process, or
    from before the stamp, whose dispatch is unknown for good."""
    if acceptance_undispatched(chat_id, str(row.get("client_message_id") or "")):
        return {"ingress_undispatched": True}
    if dispatch_entered(row):
        return {"ingress_dispatched": True}
    return {"ingress_pending": True} if accepted_here(row) else {}


def _write_marked(key: Tuple[int, str], write) -> dict:
    """A named acceptance's canonical write, under ``_INGRESS_LOCK`` before any dispatch: a raise
    marks the id undispatched (positive proof); a write that lands clears the mark."""
    try:
        row = write()
    except BaseException:
        if key[1]:
            _UNDISPATCHED.add(key)
        raise
    _UNDISPATCHED.discard(key)
    return row


def _take_undispatched(key: Tuple[int, str]) -> bool:
    """Under ``_INGRESS_LOCK``: consume that proof — True for one retry, which hands the row over."""
    found = key in _UNDISPATCHED
    _UNDISPATCHED.discard(key)
    return found


# --- which accepted row an id already names --------------------------------------------------
#
# Every web send and named skill delivery asks this before it may write, so the answer comes from a
# process-local index over the retained chat chain, never a replay of it (DEVELOPMENT 03 "Projection
# over replay"; the ``delegate_custody_memo`` fingerprint rules): the first lookup folds the whole
# chain once — the cold cost each host process pays — and later ones fold only the bytes appended
# since. The warm check is constant: the live file's identity, size and two short anchors of its
# folded prefix. A rotation keeps the folded file's identity under its archive name, so its
# remainder and any newer generation are folded next, in chain order; anything else that breaks the
# folded prefix (a shorter or rewritten file, a generation gone or moved) folds again from scratch.
# Nothing is assumed fresh: every complete row of the chain is in the index, a historical id stays
# there for good, and a chain that cannot be read raises instead of answering "absent". The index
# keeps where a row lies, not the row: a hit re-reads that one line and checks it names the id.

_ANCHOR_BYTES = 512


class _Refold(Exception):
    """The folded prefix is no longer the chain's prefix: fold the chain again from the start."""


class _Generation:
    """One folded chain file: its identity, the complete bytes folded and that prefix's anchors."""

    __slots__ = ("path", "identity", "consumed", "anchor")

    def __init__(self, path: Path, identity: tuple):
        self.path, self.identity, self.consumed, self.anchor = path, identity, 0, (b"", b"")


def _anchor(handle, consumed: int) -> tuple:
    size = min(consumed, _ANCHOR_BYTES)
    handle.seek(0)
    head = handle.read(size)
    handle.seek(consumed - size)
    return head, handle.read(size)


def _parsed(raw: bytes):
    try:  # as ``iter_jsonl_objects`` reads a line: undecodable bytes replaced, not dropped
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None


def _named_key(row) -> Optional[tuple]:
    """The ``(chat_id, client_message_id)`` an inbound row answers to, compared as the scan did."""
    if not isinstance(row, dict) or row.get("direction") != "in" or not isinstance(row.get("client_message_id"), str):
        return None
    key = (row.get("chat_id"), row["client_message_id"])
    try:
        hash(key)
    except TypeError:
        return None  # an unhashable chat_id never equals an int one
    return key


class _AcceptedIds:
    """One chat chain's index: ``(chat_id, client_message_id)`` → where its row lies.

    Its answer is the one a full scan gives: the newest generation holding the id, the first row
    naming it there. A live file ending in a complete row that lacks only its newline (a crashed
    writer's) holds it as ``unfinished``, unconsumed, until the next append completes the line.
    """

    def __init__(self, live: Path):
        self.live, self.lock, self.cold_folds = live, threading.Lock(), 0
        self._reset()

    def _reset(self) -> None:
        self.generations: list[_Generation] = []
        self.where: dict[tuple, tuple[int, int, int]] = {}  # key -> (generation, offset, length)
        self.unfinished: Optional[tuple] = None  # (key, row) at the live file's unterminated end

    def lookup(self, chat_id: int, client_message_id: str) -> Optional[dict]:
        key = (chat_id, client_message_id)
        for _attempt in range(2):
            try:
                self._advance()
                return self._row(key)
            except _Refold:
                log.debug("accepted ids of %s fold again", self.live, exc_info=True)
                self._reset()
            except BaseException:
                self._reset()  # unknown is never cached as absent: the next lookup folds again
                raise
        raise JsonlChainUnreadable(f"{self.live} kept changing while its accepted ids were folded")

    def _advance(self) -> None:
        last = self.generations[-1] if self.generations else None
        if last is not None:
            try:
                handle = self.live.open("rb")
            except FileNotFoundError:
                handle = None
            if handle is not None:
                with handle:
                    stat = os.fstat(handle.fileno())
                    if (stat.st_dev, stat.st_ino) == last.identity:
                        self._fold(len(self.generations) - 1, handle, stat, live=True)  # the warm path
                        return
        self._resync()

    def _resync(self) -> None:
        """Cold, or the live file is a new generation: walk the chain, fold what is new to it."""
        snapshot: dict = {}
        with jsonl_chain_handles(self.live, strict=True, start_offset=0, snapshot=snapshot):
            pass
        chain, known = snapshot["entries"], len(self.generations)
        if not known and chain:
            self.cold_folds += 1
        if known > len(chain):
            raise _Refold("a folded generation left the chain")
        self.unfinished = None
        for index, (path, stat, was_live) in enumerate(chain):
            identity = (stat.st_dev, stat.st_ino)
            if index < known:
                generation = self.generations[index]
                if identity != generation.identity:
                    raise _Refold("the chain's folded prefix changed")
                generation.path = path  # a rotated live file keeps its identity under its archive name
                if index < known - 1:
                    if stat.st_size != generation.consumed:
                        raise _Refold("an archive changed after it was folded")
                    continue
            else:
                self.generations.append(_Generation(path, identity))
            with path.open("rb") as handle:
                actual = os.fstat(handle.fileno())
                if (actual.st_dev, actual.st_ino) != identity:
                    raise _Refold("a generation moved while the chain was walked")
                self._fold(index, handle, actual, live=was_live and index == len(chain) - 1)

    def _fold(self, index: int, handle, stat, *, live: bool) -> None:
        generation = self.generations[index]
        if stat.st_size < generation.consumed or (
                generation.consumed and _anchor(handle, generation.consumed) != generation.anchor):
            raise _Refold(f"{generation.path} changed under its folded prefix")
        if live:
            self.unfinished = None
        handle.seek(generation.consumed)
        position = generation.consumed
        for raw in handle:
            # Only a row whose bytes hold the value "in" can be inbound: the rest is not parsed.
            key = _named_key(row := _parsed(raw)) if b'"in"' in raw else None
            if live and not raw.endswith(b"\n"):
                self.unfinished = (key, row) if key is not None else None
                break  # unconsumed: the next boundary-ensuring append completes this line
            if key is not None:
                prior = self.where.get(key)
                if prior is None or prior[0] < index:
                    self.where[key] = (index, position, len(raw))
            position += len(raw)
        if position != generation.consumed:
            generation.consumed, generation.anchor = position, _anchor(handle, position)

    def _row(self, key: tuple) -> Optional[dict]:
        located = self.where.get(key)
        newest = len(self.generations) - 1
        if self.unfinished is not None and self.unfinished[0] == key and (located is None or located[0] < newest):
            return dict(self.unfinished[1])
        if located is None:
            return None
        index, offset, length = located
        generation = self.generations[index]
        with generation.path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            if (stat.st_dev, stat.st_ino) != generation.identity:
                raise _Refold("a located row's generation moved")
            handle.seek(offset)
            row = _parsed(handle.read(length))
        if _named_key(row) != key:
            raise _Refold("a located row no longer names its id")
        return row


_ACCEPTED_IDS: dict[str, _AcceptedIds] = {}
_ACCEPTED_IDS_LOCK = threading.Lock()


def _accepted_ids(drive_root) -> _AcceptedIds:
    live = Path(drive_root) / "logs" / "chat.jsonl"
    key = os.path.abspath(live)
    with _ACCEPTED_IDS_LOCK:
        index = _ACCEPTED_IDS.get(key)
        if index is None:
            index = _ACCEPTED_IDS[key] = _AcceptedIds(live)
        return index


def reset_accepted_ids() -> None:
    """Forget every index (tests): the next lookup folds its chain again."""
    with _ACCEPTED_IDS_LOCK:
        _ACCEPTED_IDS.clear()


def accepted_chat_message(drive_root, chat_id: int, client_message_id: str) -> Optional[dict]:
    """The named inbound row ``(chat_id, client_message_id)`` already names in the retained chat chain.

    Answered by this process's index (``_AcceptedIds``); a chain that cannot be read raises
    ``OSError`` (``JsonlChainUnreadable``), never None.
    """
    index = _accepted_ids(drive_root)
    with index.lock:
        return index.lookup(chat_id, client_message_id)


def _accepted_web_message(chat_id: int, client_message_id: str) -> Optional[dict]:
    """The accepted row a web frame's own id already names (call under ``_INGRESS_LOCK``).

    Absent history is a new message; a chain that cannot be READ is unknown: the error
    refuses the frame before any claim, write or dispatch (never a second row).
    """
    from supervisor import message_bus

    if not client_message_id or message_bus.DATA_DIR is None:
        return None
    try:
        return accepted_chat_message(message_bus.DATA_DIR, chat_id, client_message_id)
    except OSError:
        log.warning("Web ingress refused id %r: retained chat is unreadable", client_message_id, exc_info=True)
        raise


def accept_local_message(bridge, drive_root, text: str, *, retain_inputs=None, dispatch=None, **message) -> tuple[dict, bool]:
    """Accept a named skill delivery once, then schedule/queue its exact source: ``(row, rejoined)``.
    The ingress lock serializes receipt-before-dispatch. A same-id retry rejoins without dispatch unless
    it takes the proof that its row never entered dispatch (``_UNDISPATCHED``): it dispatches, not
    ``rejoined``. After a crash, reads disclose a lost host session; nothing redispatches.
    """
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.chat_uploads import attachment_placeholder
    from supervisor import message_bus

    chat_id = int(message["chat_id"])
    message_id = str(message["client_message_id"])
    source = str(message["source"])
    logged = text.strip() or str(message.get("image_caption") or "").strip() or attachment_placeholder(
        message.get("image_base64"), message.get("task_metadata")
    )
    if not logged:
        raise ValueError("message is empty")
    refs = stored_refs((message.get("task_metadata") or {}).get("chat_attachments"))
    placeholder = not text.strip() and not str(message.get("image_caption") or "").strip()
    with _INGRESS_LOCK:
        row = accepted_chat_message(drive_root, chat_id, message_id)
        if row is not None:
            if row.get("source") != source:
                raise ValueError("client_message_id is already bound to another source")
            if not same_message(row, logged, refs):  # text AND ordered attachment content
                raise ValueError("client_message_id was already used for a different message")
            if not _take_undispatched((chat_id, message_id)):
                return row, True
            # Proven never dispatched: this retry's identical copies feed it; the echo shows the row's refs.
            if (message.get("task_metadata") or {}).get("chat_attachments"):
                message["task_metadata"] = {**message["task_metadata"], "chat_attachments": row.get("attachments")}
        ts = str(row.get("ts") or "") if row else utc_now_iso()
        try:
            row = row or _write_marked((chat_id, message_id), lambda: message_bus.log_chat(
                "in", chat_id, int(message.get("user_id") or 0), logged, ts=ts,
                source=source, client_message_id=message_id,
                sender_label=str(message.get("sender_label") or ""),
                transport=message.get("transport"), drive_root=drive_root, require_write=True,
                ensure_record_boundary=True,  # parseable acceptance record
                message_meta={"attachments": refs, "text_placeholder": placeholder},
            ))
        finally:
            # Once this write is attempted (or a proven-undispatched row is handed over), failure can
            # leave canonical bytes. Transfer input custody without claiming acceptance or queue
            # success; a replay/pre-write refusal never adopts this request's fresh copies.
            if retain_inputs is not None:
                retain_inputs()
        ref = build_owner_message_ref(chat_id=chat_id, client_message_id=message_id, ts=ts, text=logged)
        # record_inbound_message gets the row witness and receipt time.
        # Dispatch only schedules/queues; slow work must not hold the ingress lock.
        _enter_dispatch(row)
        (dispatch or bridge.enqueue_local_message)(text, **message, accepted_source_ref=ref, accepted_source_row=row, received_at=ts)
        return row, False


def record_inbound_message(bridge, message: dict, *, chat_id: int, user_id: int,
                           client_message_id: str, text: str, ts: str) -> Optional[dict]:
    """Keep one canonical ingress writer for dequeued and preaccepted messages."""
    from ouroboros.project_dialogue import (
        _text_sha256, build_owner_message_ref, entry_matches_source_ref, owner_message_ref_is_valid,
    )
    from supervisor import message_bus

    source = str(message.get("source") or "web")
    ref = message.get("accepted_source_ref")
    if ref:
        # A web acceptance (socket or late quiz answer) wrote this row in-process
        # before enqueue; its returned row is the item's exact witness. A skill
        # source, or a web item without a witness (the queue's empty default is
        # absent, not a row to match), re-reads the retained disk row.
        accepted_row = message.get("accepted_source_row")
        row = (accepted_row if source == "web" and isinstance(accepted_row, dict) and accepted_row
               else accepted_chat_message(message_bus.DATA_DIR, chat_id, client_message_id)) if owner_message_ref_is_valid(ref) else None
        if (not row or row.get("source") != source or row.get("chat_id") != chat_id
                or row.get("client_message_id") != client_message_id or not entry_matches_source_ref(row, [ref])
                or ref["text_sha256"] != _text_sha256(text)):
            raise ValueError("accepted source does not match the queued message")
        ref = dict(ref)
        ts = ref["ts"]
        placeholder = row.get("text_placeholder") is True
    elif message.get("suppress_chat_log"):
        return None
    else:
        metadata = message.get("task_metadata") or {}
        # no words or caption from the sender: TEXT is the host's placeholder
        placeholder = not str(message.get("text") or "").strip() and not str(message.get("image_caption") or "").strip()
        message_bus.log_chat(
            "in", chat_id, user_id, text, ts=ts, source=source,
            sender_label=str(message.get("sender_label") or ""),
            sender_session_id=str(message.get("sender_session_id") or ""),
            client_message_id=client_message_id, transport=message.get("transport"),
            client_surface=(metadata.get("client_surface") if isinstance(metadata, dict)
                            and isinstance(metadata.get("client_surface"), dict) else None),
            message_meta={"attachments": metadata.get("chat_attachments") if isinstance(metadata, dict) else None,
                          "text_placeholder": placeholder},
        )
        ref = build_owner_message_ref(chat_id=chat_id, client_message_id=client_message_id, ts=ts, text=text)
    if source != "web":
        # A parked inline photo is one of the views: one bubble, never a second photo frame;
        # its text is the row's, placeholder mark included, as history replays it.
        views = attachment_views((message.get("task_metadata") or {}).get("chat_attachments"))
        bridge.broadcast({
            "type": "photo" if message.get("image_base64") and not views else "chat", "role": "user",
            "content": text if views and placeholder else str(message.get("text") or ""),
            "caption": str(message.get("image_caption") or ""), **({"text_placeholder": True} if views and placeholder else {}),
            "image_base64": "" if views else str(message.get("image_base64") or ""), **({"attachments": views} if views else {}),
            "mime": str(message.get("image_mime") or "image/jpeg"), "ts": ts, "source": source,
            "sender_label": str(message.get("sender_label") or ""),
            "sender_session_id": str(message.get("sender_session_id") or ""),
            "client_message_id": client_message_id, "transport": message.get("transport") or {},
            "chat_id": chat_id,
        })
    return ref
