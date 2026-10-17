"""Her awareness during a live call: the room reaches the call, the call reaches the room.

A call is one turn among the room's turns, not a separate channel. While it runs,
the owner may still type into the room, another turn may answer, a task may
finish: those rows are facts the voice should know, so every row the room
accepts after the call opened becomes a ``ContextItem`` for the live session
(``RoomRows``). The reverse holds too: what was said on the call is context for
the room's RUNNING text turns, delivered through their ordinary mailbox as an
independent task's words (``UtteranceFanout``), never as an owner directive.

The room reaches the call through a post-success notification from the one
chat writer (``message_bus.log_chat`` -> ``notify_chat_row``) plus a catch-up
read of the chain for rows written between the call's admission and the
listener's registration. Rows are deduplicated by their chain address, never by
their words: two identical owner lines are two facts. The call's own rows (its
owner speech, its progress) and Main's mirror receipts of project rows are not
echoed back into the session.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional

from ouroboros.gateways.realtime_engine import ContextItem

log = logging.getLogger(__name__)

SOURCE_CALL = "call"             # the ``source`` a call stamps on the owner-speech rows it writes
ROOM_ITEM_SOURCE = "room_row"    # the ContextItem source prefix for a row of the room

_LISTENERS: Dict[int, List[Callable[[Dict[str, Any]], None]]] = {}
_LISTENERS_LOCK = threading.Lock()


def register_room_listener(chat_id: int, callback: Callable[[Dict[str, Any]], None]) -> None:
    with _LISTENERS_LOCK:
        _LISTENERS.setdefault(int(chat_id), []).append(callback)


def unregister_room_listener(chat_id: int, callback: Callable[[Dict[str, Any]], None]) -> None:
    with _LISTENERS_LOCK:
        callbacks = _LISTENERS.get(int(chat_id), [])
        if callback in callbacks:
            callbacks.remove(callback)
        if not callbacks:
            _LISTENERS.pop(int(chat_id), None)


def row_room(row: Dict[str, Any]) -> int:
    """The room a row belongs to, as the chat chain addresses it.

    0 is the hidden partition, a real room; only a missing or unreadable id is the
    legacy main-chat row (``chat_chain`` files it under 1), so the live notification
    and the catch-up read agree.
    """
    from supervisor.message_bus import coerce_chat_identity

    return coerce_chat_identity(row.get("chat_id"))


def notify_chat_row(root: Any, record: Dict[str, Any]) -> None:
    """Called by ``message_bus.log_chat`` once a row is on disk; never raises.

    The writer's own success is the only trigger: a row that was not appended
    is not a fact of the room. Listeners are copied under the lock and run
    outside it, so a slow listener cannot hold the chat writer.
    """
    chat_id = row_room(record)
    with _LISTENERS_LOCK:
        if not _LISTENERS:
            return
        callbacks = list(_LISTENERS.get(chat_id, ()))
    for callback in callbacks:
        try:
            callback(dict(record))
        except Exception:
            log.debug("room listener failed for chat %s", chat_id, exc_info=True)


def _room_member(root: pathlib.Path, chat_id: int) -> Callable[[int, Dict[str, Any]], bool]:
    """The same canonical membership ``chat_chain.iter_room_rows`` filters by."""
    from ouroboros.project_dialogue import room_membership, source_refs_for_project
    from ouroboros.projects_registry import RoomResolver, list_reserved_projects

    projects = {int(project["chat_id"]) for project in list_reserved_projects(root)}
    refs = source_refs_for_project(root, chat_id) if chat_id in projects else []
    return room_membership(chat_id, projects, refs, RoomResolver(root))


def row_key(row: Dict[str, Any]) -> str:
    """The row's chain address (chat, time, content hash): the dedupe identity."""
    from ouroboros.chat_chain import format_address, row_address

    return format_address(row_address(row))


class RoomRows:
    """New rows of one room, as ``ContextItem``s for the call that listens.

    ``start`` registers the listener FIRST, then catches up from the chain for
    rows accepted since ``started_at`` (the call's admission time), so no row
    written in between is lost and none arrives twice (the address set covers
    both paths). ``drain`` returns what accumulated since the last drain.
    """

    def __init__(self, drive_root: Any, chat_id: int, task_id: str, call_id: str, started_at: str) -> None:
        self.root = pathlib.Path(drive_root)
        self.chat_id, self.task_id, self.call_id, self.started_at = int(chat_id), str(task_id), str(call_id), started_at
        self._lock = threading.Lock()
        self._seen: set = set()
        self._pending: List[ContextItem] = []
        self._member: Optional[Callable[[int, Dict[str, Any]], bool]] = None
        self._registered = False

    def start(self) -> None:
        try:
            self._member = _room_member(self.root, self.chat_id)
        except Exception:
            log.debug("room membership for chat %s unavailable; chat id only", self.chat_id, exc_info=True)
        register_room_listener(self.chat_id, self._on_row)
        self._registered = True
        try:
            from ouroboros.chat_chain import iter_room_rows

            for _address, row, _pos in iter_room_rows(self.root, self.chat_id, period={"from": self.started_at}):
                self._on_row(row)
        except Exception:
            log.debug("room catch-up for chat %s failed; the live notifications stand", self.chat_id, exc_info=True)

    def stop(self) -> None:
        if self._registered:
            unregister_room_listener(self.chat_id, self._on_row)
            self._registered = False

    def drain(self) -> List[ContextItem]:
        with self._lock:
            items, self._pending = self._pending, []
        return items

    def accepts(self, row: Dict[str, Any]) -> bool:
        """The skip rules: the room's rows except the call's own and Main's mirrors."""
        from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES

        row_chat = row_room(row)
        if self._member is not None:
            if not self._member(row_chat, row):
                return False
        elif row_chat != self.chat_id:
            return False
        if str(row.get("task_id") or "") == self.task_id or str(row.get("source") or "") == SOURCE_CALL:
            return False
        if row.get("is_progress") or str(row.get("type") or "") in MAIN_PINNED_ROW_TYPES:
            return False
        return bool(str(row.get("text") or "").strip())

    def _on_row(self, row: Dict[str, Any]) -> None:
        if not self.accepts(row):
            return
        try:
            key = row_key(row)
        except Exception:
            return
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            self._pending.append(render_room_row(row, key))


def render_room_row(row: Dict[str, Any], key: str) -> ContextItem:
    """One room row as the mind reads it: who wrote it, then the words (bounded, disclosed)."""
    from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS
    from ouroboros.utils import truncate_within_limit

    direction = str(row.get("direction") or "")
    task_id = str(row.get("task_id") or "")
    if direction == "in":
        who = "The owner typed in this room"
    elif direction == "system":
        who = f"A system row ({row.get('type') or 'notice'}{', task ' + task_id if task_id else ''})"
    else:
        who = f"Ouroboros wrote in this room{' (turn ' + task_id + ')' if task_id else ''}"
    text = truncate_within_limit(str(row.get("text") or ""), TASK_MESSAGE_MAX_CHARS)
    return ContextItem(text=f"{who} [{key}]:\n{text}", source=f"{ROOM_ITEM_SOURCE}:{direction or 'out'}")


def running_room_turns(chat_id: int, exclude_task_id: str) -> List[str]:
    """The room's RUNNING text turns that can read a mailbox entry now.

    Direct native turns come through the census (``get_direct_chat_agent`` via
    ``direct_chat_turn``), pooled roots through the routing manifest; a
    pending task has no loop to read the entry, and the call itself is skipped.
    """
    from supervisor import workers  # the routing context: RUNNING / PENDING / DRIVE_ROOT and the census
    from ouroboros.server_routing_context import _addressable_root_tasks

    return [str(row["task_id"]) for row in _addressable_root_tasks(workers, int(chat_id))
            if row.get("status") == "running" and str(row["task_id"]) != str(exclude_task_id)]


class UtteranceFanout:
    """The call's finalized utterances, batched to the room's running text turns.

    One batch per response boundary (or per flush), split at utterance
    boundaries under ``TASK_MESSAGE_MAX_CHARS``; batch ids are deterministic
    (``call:<call_id>:<batch>:<part>``) so a resent batch is the same entry.
    Delivered as ``PROVENANCE_INDEPENDENT_TASK``: context the receiving turn
    judges, never an owner directive (only the owner supersedes an answer).
    """

    def __init__(self, drive_root: Any, chat_id: int, task_id: str, call_id: str,
                 recipients: Callable[[int, str], List[str]] = running_room_turns) -> None:
        self.root = pathlib.Path(drive_root)
        self.chat_id, self.task_id, self.call_id = int(chat_id), str(task_id), str(call_id)
        self._recipients = recipients
        self._lines: List[str] = []
        self._batches = 0
        self.delivered: List[Dict[str, Any]] = []

    def note(self, speaker: str, text: str) -> None:
        words = str(text or "").strip()
        if words:
            self._lines.append(f"{str(speaker or '').capitalize()}: {words}")

    @property
    def pending(self) -> int:
        return len(self._lines)

    def flush(self) -> int:
        """Deliver what accumulated; returns the number of mailbox entries written."""
        from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, TASK_MESSAGE_MAX_CHARS, write_task_message

        if not self._lines:
            return 0
        lines, self._lines = self._lines, []
        try:
            recipients = self._recipients(self.chat_id, self.task_id)
        except Exception:
            log.debug("call %s could not list the room's running turns", self.call_id, exc_info=True)
            recipients = []
        if not recipients:
            return 0
        self._batches += 1
        header = f"[Live call {self.call_id} in this room, spoken and transcribed]\n"
        written = 0
        for part, body in enumerate(split_at_utterances(lines, TASK_MESSAGE_MAX_CHARS - len(header)), start=1):
            msg_id = f"call:{self.call_id}:{self._batches}:{part}"
            for recipient in recipients:
                ok = write_task_message(self.root, header + body, recipient, source_task_id=self.task_id,
                                        provenance=PROVENANCE_INDEPENDENT_TASK, msg_id=msg_id)
                self.delivered.append({"task_id": recipient, "msg_id": msg_id, "ok": bool(ok)})
                written += int(bool(ok))
        return written


def split_at_utterances(lines: List[str], limit: int) -> List[str]:
    """Pack whole utterances into parts of at most ``limit`` characters.

    One utterance longer than the limit is bounded with a disclosed omission
    rather than split mid-sentence: a cut sentence would read as said.
    """
    from ouroboros.utils import truncate_within_limit

    parts: List[str] = []
    current: List[str] = []
    size = 0
    for line in lines:
        line = truncate_within_limit(line, limit)
        extra = len(line) + (1 if current else 0)
        if current and size + extra > limit:
            parts.append("\n".join(current))
            current, size = [], 0
            extra = len(line)
        current.append(line)
        size += extra
    if current:
        parts.append("\n".join(current))
    return parts
