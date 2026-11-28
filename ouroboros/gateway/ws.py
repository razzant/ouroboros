"""WebSocket gateway dispatch and broadcast state."""

from __future__ import annotations

import asyncio
import collections
import functools
import inspect
import itertools
import json
import logging
import pathlib
import re
import threading
import time
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros import chat_uploads
from ouroboros.asgi_beat import ASGI_BEAT
from ouroboros.config import DATA_DIR
from ouroboros.gateway._helpers import run_sync_to_completion, settle_to_completion
from ouroboros.gateway.completion_gist import short_gist_frame
from ouroboros.process_logging import RecordHandoff
from ouroboros.utils import json_dumps_utf8, utc_now_iso

log = logging.getLogger(__name__)
# The socket lifecycle lines this module logs on the ASGI loop (connect, disconnect, send failure, receive
# errors) leave it through ONE ordered, bounded handoff: a slow log sink never holds a heartbeat, a receive
# or Panic. Lines keep their creation time; a kill loses the few the writer has not reached.
_LIFECYCLE_LOG = RecordHandoff("ws-lifecycle-log")

# The close code of the browser's silence watchdog (``web/modules/ws.js``): an application code
# (RFC 6455 4000-4999) whose reason carries only the fields below, each a bounded integer.
WATCHDOG_CLOSE_CODE = 4408
_WATCHDOG_REASON_FIELDS = {"silent_ms": re.compile(r"[0-9]{1,10}"), "tick_gap_ms": re.compile(r"[0-9]{1,10}"),
                           "hidden": re.compile(r"[01]")}
_SERIALS = itertools.count(1)


class _Connection:
    """One open browser socket and its ONE writer.

    Broadcast frames reach the socket in order through its own outbox, served by one writer
    task for the socket's whole life: a frame is never a coroutine of its own, and a slow or
    half-open socket delays only itself (WS4). A send that fails retires the connection (it
    leaves the registry; its receive loop ends on its own) and its diagnostic row is written
    off the loop. Replies to the socket's own frames (notices, ``panic_accepted``) are sent by
    its receive loop directly, as before.

    For its one disconnect line (``_close_facts``) it keeps, in memory only: an ephemeral
    ``serial`` (this process's count), when it opened, when its writer last completed a send and
    of which kind (``heartbeat`` or ``other``), since when a send is being awaited, the largest
    ASGI loop lateness it lived through (``note_loop_lateness``) and a ``cause`` only the server
    knows (its own close, a write error: the first one named stays).
    """

    __slots__ = ("ws", "outbox", "writer", "profile_scope", "serial", "opened", "sent", "sent_kind", "sending",
                 "beat_late_max", "cause")

    def __init__(self, ws: Any, profile_scope: str = "") -> None:
        self.ws = ws
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.writer = asyncio.get_running_loop().create_task(self._write(), name="ws-writer")
        # The serving generation's ONE frozen data-profile scope (ServedUI.apply at admission,
        # via ServedUI.serving_profile_scope); "" when no generation was applied. Live chat
        # admission compares a frame's claim against it (#3294).
        self.profile_scope = profile_scope
        self.serial, self.opened = next(_SERIALS), time.monotonic()
        self.sent: float | None = None
        self.sent_kind = ""
        self.sending: float | None = None
        self.beat_late_max = 0.0
        self.cause = ""

    async def _write(self) -> None:
        while True:
            data, msg_type = await self.outbox.get()
            self.sending = time.monotonic()
            try:
                await self.ws.send_text(data)
            except Exception as exc:
                self.sending = None
                _retire(self, msg_type, exc)
                return
            self.sent, self.sending = time.monotonic(), None
            self.sent_kind = "heartbeat" if msg_type == "heartbeat" else "other"


_ws_clients: list[_Connection] = []
_ws_lock = threading.Lock()
_event_loop: asyncio.AbstractEventLoop | None = None
# Frames encoded off the loop and not yet handed to the writers: one loop wake per burst.
_pending: collections.deque = collections.deque()
_flush_scheduled = False


def set_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Set the server event loop used by ``broadcast_ws_sync``."""
    global _event_loop
    _event_loop = loop


def _chat_attachment_uploads(attachments: Any) -> list[dict]:
    """Resolve EVERY desktop-chat attachment (any type) to a staging spec.

    v6.52.0 (P1, full desktop unify): the paperclip allows multiple files of any
    type. Each references a file already stored by /api/chat/upload under
    data/uploads/, addressed by a validated basename (no traversal); a missing or
    invalid one keeps its ordinal with an empty path (rejected at staging), and
    ``_accept_with_attachments`` binds every spec to its measured identity. The
    returned specs feed ``stage_task_attachments`` in the worker, which routes the
    WHOLE set through the shared artifact_store substrate (images natively + every
    other file via the read_file manifest), so nothing is silently dropped.

    Returns a list of ``{"path": <abs uploads path>, "label": <display>, "mime"}``.
    """
    if not isinstance(attachments, list):
        return []
    import os as _os

    uploads_dir = chat_uploads.uploads_dir(DATA_DIR).resolve(strict=False)
    specs: list[dict] = []
    for ordinal, item in enumerate(attachments):
        if not isinstance(item, dict):
            specs.append({"path": "", "label": f"attachment {ordinal + 1}", "mime": ""})
            continue
        name = _os.path.basename(str(item.get("filename") or "").strip())
        label = str(item.get("display_name") or item.get("label") or name).strip()
        if not name or name in {".", ".."}:
            specs.append({"path": "", "label": label or f"attachment {ordinal + 1}", "mime": str(item.get("mime") or "")})
            continue
        path = (uploads_dir / name).resolve(strict=False)
        try:
            path.relative_to(uploads_dir)
        except ValueError:
            specs.append({"path": "", "label": label or name, "mime": str(item.get("mime") or "")})
            continue
        if not path.is_file():
            specs.append({"path": "", "label": label or name, "mime": str(item.get("mime") or "")})
            continue
        specs.append({"path": str(path), "label": label or name, "mime": str(item.get("mime") or "")})
    return specs


def _accept_with_attachments(bridge: Any, payload: str, send_kwargs: dict, attachments: Any) -> None:
    """Measure a web frame's attachments off the loop, then accept the message.

    The refs ride ``task_metadata["chat_attachments"]`` (the client_surface rail) to
    the canonical writer, which claims them and records them on the inbound row.
    Each model staging spec (one per frame item, in order) carries its ref's measured
    identity, which ``stage_task_attachments`` verifies while copying: a path swapped
    after this measurement stages nothing else. An unavailable ref stages nothing.
    """
    uploads = _chat_attachment_uploads(attachments)
    refs = chat_uploads.refs_for_frame(attachments, DATA_DIR)
    for spec, ref in zip(uploads, refs):
        spec.update({"path": ""} if ref.get("unavailable") else {"size": ref["size"], "sha256": ref["sha256"]})
    send_kwargs["task_metadata"]["chat_attachment_uploads"] = uploads
    send_kwargs["task_metadata"]["chat_attachments"] = refs
    bridge.ui_send(payload, **send_kwargs)


class _RoomWordLapsed(RuntimeError):
    """The core no longer confirms the room a Send again stood on: nothing of it was saved or handed over. Its notice
    names it by ``reason_code`` (the typed fact a page acts on, never the notice's words)."""

    reason_code = "ordinary_room_unconfirmed"


def _accept_on_room_word(accept, chat_id: int, named: Any, claimed: str, serving: str) -> None:
    """Accept a Send again of a copy the page showed on the core's word that this room is one of its profile's
    ordinary rooms (``ChatInbound.ordinary_room``) only while that word still holds.

    It runs in the ordered acceptance, off the loop, before anything is measured, saved or handed over: the frame names
    this room, its page's ``profile_scope`` is this connection's own proven one, and ``room_archive.ordinary_room``
    still counts the room in the data root the canonical writer uses (``message_bus.DATA_DIR``). Otherwise
    ``_RoomWordLapsed`` answers the sender and nothing is written. Only such a frame comes here, ordinary frames never
    do. Panic and commands never wait for it; the chat frames after it on the same socket settle behind it in their
    order, as behind any acceptance.
    """
    from ouroboros.room_archive import ordinary_room
    from supervisor import message_bus

    root = message_bus.DATA_DIR
    if not (serving and claimed == serving and type(named) is int and named == chat_id
            and root is not None and ordinary_room(root, chat_id)):
        raise _RoomWordLapsed(f"room {chat_id} is not confirmed as one of this profile's ordinary rooms")
    accept()


def has_ws_clients() -> bool:
    with _ws_lock:
        return bool(_ws_clients)


def note_loop_lateness(late_sec: float) -> None:
    """On the loop, at every heartbeat (``server_runtime.ws_heartbeat_loop``): each open connection keeps the
    largest ASGI loop lateness it lived through, so a gap survives until its disconnect line reports it."""
    with _ws_lock:
        connections = list(_ws_clients)
    for connection in connections:
        connection.beat_late_max = max(connection.beat_late_max, late_sec)


async def close_all_ws(*, code: int = 1012, reason: str = "Server restarting") -> None:
    """Close every connected browser websocket best-effort; each disconnect line names the server's close."""
    with _ws_lock:
        connections = list(_ws_clients)
    for connection in connections:
        connection.cause = f"server_close:{code}"
        try:
            await connection.ws.close(code=code, reason=reason)
        except Exception:
            pass


def _register(ws: Any, profile_scope: str = "") -> _Connection:
    """Queue the connection's own scope before listing it for broadcasts, without awaiting a send."""
    connection = _Connection(ws, profile_scope)
    if profile_scope:
        frame = json.dumps({"type": "connection_profile", "profile_scope": profile_scope})
        connection.outbox.put_nowait((frame, "connection_profile"))
    with _ws_lock:
        _ws_clients.append(connection)
    return connection


def _profile_scope_mismatch(serving_scope: str, claimed: str) -> bool:
    """True only on POSITIVE proof that a frame's page belongs to another data profile: the serving
    connection's scope and the frame's claim are both proven (non-empty) and differ. An unproven
    serving connection, or a frame that carries no claim (a legacy page), is no proof and admits;
    Panic and the product controls never pass through this check at all (#3294)."""
    return bool(serving_scope) and bool(claimed) and claimed != serving_scope


def _unregister(connection: _Connection) -> int:
    with _ws_lock:
        if connection in _ws_clients:
            _ws_clients.remove(connection)
        total = len(_ws_clients)
    connection.writer.cancel()
    return total


def _retire(connection: _Connection, msg_type: str, exc: BaseException) -> None:
    """A writer whose send failed: drop the connection and record the fact OFF the loop.

    The ``broadcast_partial_failure`` row keeps its shape (one dead client per failed writer);
    it used to be appended synchronously inside the broadcast, a file lock and a write on the
    event loop. The default executor takes it now; a full executor only delays the row. The
    connection's disconnect line, written when its receive loop ends, names the write error, unless
    the server's own close was named first (a send failing after it does not replace it). The dead
    client leaves the registry before its line is logged, so the line's count is the new one."""
    if not connection.cause:
        connection.cause = f"write_error:{type(exc).__name__[:64]}"
    with _ws_lock:
        total_clients = len(_ws_clients)
        if connection in _ws_clients:
            _ws_clients.remove(connection)
        remaining = len(_ws_clients)
    _LIFECYCLE_LOG.log(log, logging.INFO, "WebSocket send failed for msg type=%s; dropped client %d (%s) (total: %d)",
                       msg_type, connection.serial, type(exc).__name__, remaining)
    row = {"ts": utc_now_iso(), "type": "broadcast_partial_failure", "msg_type": msg_type,
           "dead_clients": 1, "total_clients": total_clients}
    try:
        asyncio.get_running_loop().run_in_executor(None, _record_failure, pathlib.Path(DATA_DIR), row)
    except Exception:
        _LIFECYCLE_LOG.log(log, logging.DEBUG, "Failed to hand off broadcast_partial_failure", exc_info=True)


def _record_failure(data_dir: pathlib.Path, row: dict) -> None:
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(data_dir / "logs" / "events.jsonl", row)
    except Exception:
        log.debug("Failed to emit broadcast_partial_failure event", exc_info=True)


def _encode(msg: dict) -> str:
    return json_dumps_utf8(short_gist_frame(msg), ensure_ascii=False, default=str)


def _queue_frame(msg: dict) -> bool:
    """Encode one frame and queue it for the writers; returns whether the caller must schedule
    the flush (False: no socket is open, or a flush is already scheduled).

    Nothing is encoded for nobody: a frame with no open socket costs only this check (cost
    law L10; every ``log`` frame used to be JSON-encoded on the loop with no browser open)."""
    global _flush_scheduled
    with _ws_lock:
        if not _ws_clients:
            return False
    frame = (_encode(msg), str(msg.get("type", "unknown")))
    with _ws_lock:
        _pending.append(frame)
        schedule, _flush_scheduled = not _flush_scheduled, True
    return schedule


def _flush_pending() -> None:
    """On the loop: hand every queued frame, in order, to every open connection's writer."""
    global _flush_scheduled
    with _ws_lock:
        frames, connections = list(_pending), list(_ws_clients)
        _pending.clear()
        _flush_scheduled = False
    for connection in connections:
        for frame in frames:
            connection.outbox.put_nowait(frame)


async def broadcast_ws(msg: dict) -> None:
    """Hand one frame to every open connection's writer (on the loop), behind every frame
    queued before it; the writers send. A frame with no open socket is not even encoded."""
    _queue_frame(msg)
    _flush_pending()


def broadcast_ws_sync(msg: dict) -> bool:
    """Thread-safe broadcast. ``True`` when a running server loop took the frame (an open socket
    or none — a frame for nobody is dropped unencoded); ``False`` in a process without one (a
    pooled worker) or once the loop stopped. The frame is encoded on the caller's thread and one
    ``call_soon_threadsafe`` flush carries a whole burst to the writers: no coroutine per frame,
    and a stopped or closed loop never holds an un-awaited one."""
    global _flush_scheduled
    loop = _event_loop
    if loop is None or not loop.is_running():
        return False
    if not _queue_frame(msg):
        return True
    try:
        loop.call_soon_threadsafe(_flush_pending)
    except RuntimeError:  # the loop closed after the check: nothing will flush these frames
        with _ws_lock:
            _pending.clear()
            _flush_scheduled = False
        return False
    return True


def notify_extension_health_changed() -> None:
    """Ask Skills to reread server health or a newly unconfirmed companion stop."""
    broadcast_ws_sync({"type": "extension_health_changed"})


async def _dispatch_extension_message(
    websocket: WebSocket,
    msg: dict[str, Any],
    msg_type: str,
) -> bool:
    """Return True when an extension handler owned this message."""
    parsed_ext_type = None
    if isinstance(msg_type, str):
        try:
            from ouroboros.extension_loader import parse_extension_surface_name

            parsed_ext_type = parse_extension_surface_name(msg_type)
        except Exception:
            parsed_ext_type = None
    if not parsed_ext_type:
        return False

    state = None
    try:
        from ouroboros.config import get_skills_repo_path, load_settings
        from ouroboros.extension_loader import (
            extension_name_prefix,
            list_ws_handlers,
            reconcile_extension,
            runtime_state_for_skill_name,
        )
        from ouroboros.skill_peer_inventory import discover_skill_peers

        drive_root = pathlib.Path(
            websocket.app.state.drive_root  # type: ignore[attr-defined]
            if hasattr(websocket.app, "state") and hasattr(websocket.app.state, "drive_root")
            else DATA_DIR
        )
        repo_dir = pathlib.Path(
            websocket.app.state.repo_dir  # type: ignore[attr-defined]
            if hasattr(websocket.app, "state") and hasattr(websocket.app.state, "repo_dir")
            else pathlib.Path(__file__).resolve().parents[2]
        )
        repo_path = get_skills_repo_path()
        handler_spec = list_ws_handlers().get(msg_type)
        skill_name = str((handler_spec or {}).get("skill") or "")
        if not skill_name:
            for skill in await asyncio.to_thread(discover_skill_peers, drive_root, repo_path=repo_path):
                if msg_type.startswith(extension_name_prefix(skill.name)):
                    skill_name = skill.name
                    break
        if not skill_name:
            raise KeyError(msg_type)
        state = await asyncio.to_thread(reconcile_extension, skill_name, drive_root, load_settings, repo_path=repo_path)
        state = {**state, **await asyncio.to_thread(runtime_state_for_skill_name, skill_name, drive_root, repo_path=repo_path)}
        if not state.get("desired_live"):
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"extension WS handler {msg_type!r} is not live: {state.get('reason')}"}}))
            return True
        if state.get("action") == "extension_load_error" or not state.get("live_loaded"):
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"extension WS handler {msg_type!r} failed to go live: {state.get('load_error') or state.get('reason')}"}}))
            return True
        handler_spec = list_ws_handlers().get(msg_type)
    except Exception:
        handler_spec = None

    if handler_spec is None:
        extra = ""
        if isinstance(state, dict) and state.get("action") == "extension_load_error":
            extra = f" (load_error={state.get('load_error')})"
        await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"no extension WS handler for {msg_type!r}{extra}"}}))
        return True

    if handler_spec.get("out_of_process"):
        from ouroboros.deployment_extensions import failure_words
        try:
            from ouroboros.extension_process_runner import dispatch_extension_ws_subprocess

            result = await asyncio.to_thread(
                dispatch_extension_ws_subprocess,
                handler_spec,
                msg,
                drive_root=drive_root,
                repo_dir=repo_dir,
            )
            if result is not None:
                await websocket.send_text(json.dumps({"type": msg_type + ".reply", "data": result}))
        except Exception as exc:
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "error", "message": f"extension WS handler {msg_type!r} child failed: {failure_words(skill_name, exc)}"}}))
        return True

    handler = handler_spec.get("handler")
    from ouroboros.deployment_extensions import failure_words
    try:
        from ouroboros.extension_process_runner import disclose_inprocess_extension_dispatch

        # A ledger append under the money lock: off the ASGI loop, as the HTTP path does.
        await asyncio.to_thread(
            disclose_inprocess_extension_dispatch,
            handler_spec,
            drive_root=drive_root,
            surface_kind="ws",
            surface=msg_type,
        )
    except Exception as exc:
        await websocket.send_text(json.dumps({
            "type": "log",
            "data": {
                "level": "error",
                "message": (
                    f"extension WS handler {msg_type!r} model-cost disclosure failed: "
                    f"{failure_words(skill_name, exc)}"
                ),
            },
        }))
        return True
    try:
        # Mirror the HTTP dispatcher: a synchronous in-process handler (and its
        # synchronous barrier wait) runs off the ASGI loop, so one skill's
        # blocking callback never stalls unrelated HTTP and WebSocket work.
        if not callable(handler):
            result = None
        elif inspect.iscoroutinefunction(handler):
            result = await handler(msg)
        else:
            result = await asyncio.to_thread(handler, msg)
        if inspect.iscoroutine(result):
            result = await result
        if result is not None:
            await websocket.send_text(json.dumps({"type": msg_type + ".reply", "data": result}))
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "log", "data": {"level": "error", "message": f"extension WS handler {msg_type!r} raised: {failure_words(skill_name, exc)}"}}))
    return True


def _initialization_notice(client_message_id: str = "", chat_id: int = 1, *, reason: str = "",
                           reason_code: str = "") -> str:
    """The refusal of this socket's frame; it names that frame so only its Sending settles.
    ``reason`` says why the frame was not accepted, without exposing internal errors; ``reason_code``, given only for
    a refusal a page must tell apart (``_RoomWordLapsed``, ``message_ingress.ModelChoiceChanged``), is that typed cause."""
    return json.dumps({
        "type": "chat", "chat_id": chat_id,
        "role": "system", "system_type": "initialization_notice",
        "content": reason or "⚠️ System is still initializing. Please wait a moment and try again.",
        "ts": utc_now_iso(),
        **({"client_message_id": client_message_id} if client_message_id else {}),
        **({"reason_code": reason_code} if reason_code else {}),
    })


def _no_bridge_notice(websocket: WebSocket, client_message_id: str, chat_id: int) -> str:
    """The notice for a chat frame that arrived before the dialogue opened."""
    from ouroboros.server_liveness import boot_notice

    return _initialization_notice(client_message_id, chat_id,
                                  reason=boot_notice(getattr(websocket.app, "state", None)))


async def _accept_chat_after(websocket: WebSocket, previous: asyncio.Task | None, accept,
                             client_message_id: str = "", chat_id: int = 1) -> None:
    """Run one chat frame's acceptance once this socket's previous one settled.

    The web acceptance takes the single host's ingress lock and a locked durable
    append (log_chat(require_write=True)); either may wait behind a skill delivery
    scanning retained chat or a slow disk, so it runs off the ASGI loop, and the
    receive loop never awaits it: a command frame on the same socket (Panic,
    Restart) is admitted meanwhile. ``run_sync_to_completion`` keeps custody of
    row → queue → echo; a failure answers the sender with the initialization notice.
    """
    if previous is not None:
        await asyncio.wait((previous,))  # its failure was its own notice
    try:
        await run_sync_to_completion(accept)
    except Exception as exc:
        from ouroboros.chat_references import ReplyExcerptNotAuthored, ReplyExcerptUnavailable
        from ouroboros.store import StoreBusy
        from supervisor.message_ingress import ClientMessageIdConflict, ModelChoiceChanged

        busy = isinstance(exc, StoreBusy) or isinstance(exc.__cause__, StoreBusy)
        reason = "⚠️ Your message was not saved: the dialogue store is busy. Please send it again." if busy else ""
        code = ""
        if isinstance(exc, ReplyExcerptUnavailable):
            reason = "The quoted message is unavailable. Retry or clear the quote."
        elif isinstance(exc, ReplyExcerptNotAuthored):
            reason = "This message has no text you can quote. Clear the quote or select another passage."
        elif isinstance(exc, ClientMessageIdConflict):
            reason = "This send already belongs to a different message. Send these words as a new message."
        elif isinstance(exc, ModelChoiceChanged):
            reason, code = str(exc), exc.reason_code
        elif isinstance(exc, _RoomWordLapsed):
            reason = ("⚠️ Your message was not saved: Ouroboros could not confirm this room just now. "
                      "This tab keeps it.")
            code = exc.reason_code
        try:
            await websocket.send_text(_initialization_notice(client_message_id, chat_id, reason=reason,
                                                             reason_code=code))
        except Exception:
            _LIFECYCLE_LOG.log(log, logging.DEBUG, "WebSocket closed before its chat failure notice", exc_info=True)


def watchdog_reason_fields(reason: Any) -> list[str]:
    """The fields the browser's watchdog puts in its close reason (``silent_ms=<n> tick_gap_ms=<n> hidden=<0|1>``),
    parsed strictly: only those keys, each once, with a bounded integer (``hidden``: 0 or 1). Anything else in the
    reason is dropped, never echoed; the reason itself is at most 123 bytes (RFC 6455)."""
    fields: dict[str, str] = {}
    for token in str(reason or "").split():
        key, sep, value = token.partition("=")
        pattern = _WATCHDOG_REASON_FIELDS.get(key)
        if sep and pattern is not None and key not in fields and pattern.fullmatch(value):
            fields[key] = value
    return [f"{key}={fields[key]}" for key in _WATCHDOG_REASON_FIELDS if key in fields]


def _close_facts(connection: _Connection, code: int | None, reason: Any) -> str:
    """One disconnect's facts, in a bounded line: the ASGI close code as received (no code at all is ``unknown``;
    1005 and 1006 stay what they are, never read as an initiator), the watchdog's own fields when the code is
    ours, and what the connection kept: age, last completed send and its kind, the age of a send still awaited,
    the outbox depth, the largest loop lateness it lived through (a beat overdue right now included) and a cause
    only the server knows."""
    now = time.monotonic()
    facts = [f"conn={connection.serial}", f"code={'unknown' if code is None else code}"]
    if code == WATCHDOG_CLOSE_CODE:
        facts += watchdog_reason_fields(reason)
    facts += [f"age={now - connection.opened:.1f}s",
              f"last_send={connection.sent_kind}:{now - connection.sent:.1f}s" if connection.sent is not None
              else "last_send=none",
              f"in_flight={now - connection.sending:.1f}s" if connection.sending is not None else "in_flight=none",
              f"outbox={connection.outbox.qsize()}",
              f"beat_late_max={max(connection.beat_late_max, ASGI_BEAT.overdue(now)):.1f}s"]
    if connection.cause:
        facts.append(f"cause={connection.cause}")
    return " ".join(facts)


async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    # This connection's data-profile scope is the serving generation's ONE scope, frozen at
    # admission (ServedUI.apply, after the data root is admitted) and read here from memory:
    # no disk I/O and no awaited read before the receive loop, so Panic and the control path
    # never wait behind it. "" only on an app that never applied a generation: no proof,
    # admission stays as it was (#3294).
    served_ui = getattr(getattr(websocket.app, "state", None), "served_ui", None)
    profile_scope = served_ui.serving_profile_scope() if served_ui is not None else ""
    connection = _register(websocket, profile_scope)
    with _ws_lock:
        total = len(_ws_clients)
    _LIFECYCLE_LOG.log(log, logging.INFO, "WebSocket client connected (total: %d) conn=%d", total, connection.serial)
    # Tail of this socket's chat acceptances: each waits for the one before it,
    # so its chat frames settle in receive order while the loop keeps receiving.
    accepting: asyncio.Task | None = None
    controls: set[asyncio.Task] = set()
    close_code: int | None = None  # the ASGI disconnect's code; None: the loop ended without one
    close_reason = ""
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue

            msg_type = str(msg.get("type", "") or "")
            if await _dispatch_extension_message(websocket, msg, msg_type):
                continue

            # Executable gateway ABI (ABI-3, Q7=A): inbound chat/command frames
            # are validated against the derived contract schema at THIS ingress
            # seam only — egress and history replay are never validated.
            if msg_type in ("chat", "command"):
                from ouroboros.gateway.contracts import ChatInbound, CommandInbound
                from ouroboros.gateway.schema import validate_ingress

                schema_errors = validate_ingress(
                    msg, ChatInbound if msg_type == "chat" else CommandInbound)
                if schema_errors:
                    await websocket.send_text(json.dumps({
                        "type": "log",
                        "data": {
                            "level": "warning",
                            "message": ("ingress schema rejected a "
                                        f"{msg_type} message: "
                                        + "; ".join(schema_errors[:5])),
                        },
                    }))
                    continue

            payload = msg.get("content", "") if msg_type == "chat" else msg.get("cmd", "")
            if msg_type in ("chat", "command") and (payload or (msg_type == "chat" and msg.get("attachments"))):

                try:
                    thread_id = int(msg.get("chat_id") or 1) if msg_type == "chat" else 1
                except (TypeError, ValueError):
                    thread_id = 1
                try:
                    from ouroboros.client_surface import normalize_client_surface
                    from supervisor.message_bus import try_get_bridge

                    bridge = try_get_bridge()
                    if str(payload).strip().lower().startswith("/panic"):
                        from ouroboros.panic_source import source_evidence
                        from ouroboros.server_control import request_panic
                        evidence = source_evidence(websocket, "chat_command" if msg_type == "chat" else "web_ui_ws",
                                                   client_surface=msg.get("client_surface"))
                        if request_panic(bridge, payload, evidence=evidence,
                                         startup_owner_command=getattr(websocket.app.state, "startup_owner_command", None)):
                            await websocket.send_text(json.dumps({"type": "panic_accepted"}))
                        else:
                            await websocket.send_text(json.dumps({"type": "log", "data": {
                                "level": "warning", "message": "Panic emergency owner is unavailable."}}))
                        continue  # Never enter ordered chat acceptance or ordinary command admission.
                    if msg_type == "chat":
                        # Live admission is bound to this serving connection's confirmed profile:
                        # a frame whose page proves a DIFFERENT profile is not saved here (#3294).
                        # Panic above and the command controls below never pass through this gate.
                        claimed_scope = str(msg.get("profile_scope") or "")
                        if _profile_scope_mismatch(connection.profile_scope, claimed_scope):
                            await websocket.send_text(_initialization_notice(
                                str(msg.get("client_message_id") or ""), thread_id,
                                reason="⚠️ This page was opened under a different data profile; the message "
                                       "was not saved. Reload the page to continue."))
                            continue
                        client_surface = normalize_client_surface(msg.get("client_surface"))
                        task_metadata: dict[str, Any] = {
                            "reply_to": msg.get("reply_to", ""),
                            **({"reply_excerpt": msg["reply_excerpt"]} if "reply_excerpt" in msg else {}),
                            # Sent as new work: only the condition its acceptance compares, never a model.
                            **({"new_work_token": msg["new_work"]} if msg.get("new_work") else {}),
                        }
                        # Attachment inspection belongs to the ordered acceptance
                        # worker, so this socket can receive Panic during disk I/O.
                        if client_surface is not None:
                            # Sending-surface observables ride task_metadata
                            # through the bus whitelist
                            # nested, reach direct/ephemeral/queued turns, and
                            # land in chat.jsonl at the canonical-row writer.
                            client_surface["received_at"] = utc_now_iso()
                            task_metadata["client_surface"] = client_surface
                        send_kwargs = dict(
                            broadcast=True,
                            sender_session_id=str(msg.get("sender_session_id", "") or ""),
                            client_message_id=str(msg.get("client_message_id", "") or ""),
                            task_metadata=task_metadata,
                            chat_id=thread_id,
                            project_id=str(msg.get("project_id", "") or ""),
                        )
                        if str(payload).strip().lower() == "/restart":
                            from ouroboros.server_control import dispatch_accepted_restart

                            send_kwargs["dispatch"] = functools.partial(
                                dispatch_accepted_restart, bridge,
                                callback=getattr(websocket.app.state, "startup_owner_command", None))
                    else:
                        from supervisor.message_ingress import ui_control_fact

                        try:  # a control's press composed this command: no row will speak as my human
                            control = ui_control_fact(msg.get("ui_control"), pressed_at=utc_now_iso(), command=str(payload))
                        except ValueError as exc:
                            await websocket.send_text(json.dumps({"type": "log", "data": {
                                "level": "warning", "message": f"command refused: {exc}"}}))
                            continue
                        send_kwargs = {"broadcast": False, **({"task_metadata": {"ui_control": control}} if control else {})}
                    if msg_type == "command" and (bridge is None or str(payload).strip().lower() == "/restart"):
                        callback = getattr(websocket.app.state, "startup_owner_command", None)
                        action = callback(payload, send_kwargs=send_kwargs) if callable(callback) else None
                        if action is not None:
                            # A slow checkout cannot hold this socket's Panic behind
                            # Restart. Retain accepted control work through disconnect.
                            control = asyncio.create_task(_accept_chat_after(
                                websocket, None, action, str(send_kwargs.get("client_message_id") or ""), thread_id))
                            controls.add(control)
                            control.add_done_callback(controls.discard)
                            continue
                    if bridge is None:  # the store is still migrating (or the start failed): say so
                        await websocket.send_text(_no_bridge_notice(
                            websocket, str(msg.get("client_message_id") or ""), thread_id))
                        continue
                    if msg_type == "chat":
                        accept = (functools.partial(_accept_with_attachments, bridge, payload, send_kwargs,
                                                    msg["attachments"]) if msg.get("attachments")
                                  else functools.partial(bridge.ui_send, payload, **send_kwargs))
                        if msg.get("ordinary_room") is not None:  # a Send again on the core's word: asked again first
                            accept = functools.partial(_accept_on_room_word, accept, thread_id,
                                                       msg.get("ordinary_room"), claimed_scope,
                                                       connection.profile_scope)
                        accepting = asyncio.create_task(_accept_chat_after(
                            websocket, accepting, accept, send_kwargs["client_message_id"], thread_id))
                    else:
                        bridge.ui_send(payload, **send_kwargs)
                except Exception:
                    await websocket.send_text(_initialization_notice(str(msg.get("client_message_id") or ""), thread_id))
    except WebSocketDisconnect as exc:
        close_code, close_reason = (exc.code if type(exc.code) is int else None), exc.reason
    except Exception as exc:
        _LIFECYCLE_LOG.log(log, logging.WARNING, "WebSocket error: %s", exc)
    finally:
        total = _unregister(connection)
        _LIFECYCLE_LOG.log(log, logging.INFO, "WebSocket client disconnected (total: %d) %s", total,
                           _close_facts(connection, close_code, close_reason))
        if accepting is not None:
            # Received chat frames keep custody through disconnect or cancellation:
            # the socket task returns only after each settled row → queue → echo.
            await settle_to_completion(accepting)
        if controls:
            await settle_to_completion(asyncio.gather(*controls))


__all__ = [
    "WATCHDOG_CLOSE_CODE",
    "broadcast_ws",
    "broadcast_ws_sync",
    "close_all_ws",
    "has_ws_clients",
    "note_loop_lateness",
    "notify_extension_health_changed",
    "set_event_loop",
    "watchdog_reason_fields",
    "ws_endpoint",
]
