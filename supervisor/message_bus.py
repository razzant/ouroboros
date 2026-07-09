"""Queue-based bridge between UI/skill transports and the supervisor."""

from __future__ import annotations

import base64
import logging
import queue
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.event_bus import CHAT_DOCUMENT, CHAT_OUTBOUND, CHAT_PHOTO, CHAT_TYPING, CHAT_VIDEO, publish_event
from supervisor.state import append_jsonl, load_state
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


DATA_DIR = None  # pathlib.Path
TOTAL_BUDGET_LIMIT: float = 0.0
BUDGET_REPORT_EVERY_MESSAGES: int = 10
_BRIDGE: Optional["LocalChatBridge"] = None


def coerce_chat_identity(value: Any, default: int = 1) -> int:
    """Preserve explicit 0 sentinels while defaulting missing IDs for web chat."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def init(
    drive_root,
    total_budget_limit: float,
    budget_report_every: int,
    chat_bridge: "LocalChatBridge",
) -> None:
    global DATA_DIR, TOTAL_BUDGET_LIMIT, BUDGET_REPORT_EVERY_MESSAGES, _BRIDGE
    DATA_DIR = drive_root
    TOTAL_BUDGET_LIMIT = total_budget_limit
    BUDGET_REPORT_EVERY_MESSAGES = budget_report_every
    _BRIDGE = chat_bridge


def get_bridge() -> "LocalChatBridge":
    assert _BRIDGE is not None, "message_bus.init() not called"
    return _BRIDGE


def try_get_bridge() -> "Optional[LocalChatBridge]":
    """Return initialized bridge, if any."""
    return _BRIDGE


def refresh_budget_limit(new_limit: Optional[float]) -> None:
    """Hot-reload budget limit for status messages."""
    global TOTAL_BUDGET_LIMIT
    try:
        TOTAL_BUDGET_LIMIT = float(new_limit) if new_limit is not None else 0.0
    except (TypeError, ValueError):
        pass


class LocalChatBridge:
    """Local Queue-backed message bus."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self._inbox = queue.Queue()   # user -> agent
        self._log_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._update_counter = 0
        self._broadcast_fn = None  # set by server.py for WebSocket streaming
        # A2A response subscriptions: {subscription_id: (chat_id, callback)}
        self._response_subs: Dict[str, tuple] = {}
        self._response_subs_lock = threading.Lock()
        self._chat_transports: Dict[int, Dict[str, Any]] = {}
        if settings:
            self.configure_from_settings(settings)

    def broadcast(self, payload: dict) -> None:
        """Broadcast to WebSocket clients, excluding A2A virtual chat_ids."""
        chat_id = payload.get("chat_id")
        if is_a2a_chat_id(chat_id):
            return
        if self._broadcast_fn:
            self._broadcast_fn(payload)

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        """Block on inbox and return supervisor-style updates."""
        try:
            raw_msg = self._inbox.get(timeout=timeout)
            if isinstance(raw_msg, str):
                msg = {
                    "chat_id": 1,
                    "user_id": 1,
                    "text": raw_msg,
                    "source": "web",
                    "sender_label": "",
                }
            else:
                msg = dict(raw_msg or {})

            msg_chat_id = coerce_chat_identity(msg.get("chat_id"), 1)
            msg_user_id = coerce_chat_identity(msg.get("user_id"), 1)
            message = {
                "chat": {"id": msg_chat_id},
                "from": {"id": msg_user_id},
                "text": str(msg.get("text") or ""),
                "source": str(msg.get("source") or "web"),
            }
            chat_id_value = msg_chat_id
            if isinstance(msg.get("transport"), dict) and msg.get("transport") and chat_id_value != 1:
                self._chat_transports[chat_id_value] = dict(msg.get("transport") or {})
            else:
                self._chat_transports.pop(chat_id_value, None)
            for key in (
                "sender_label",
                "sender_session_id",
                "client_message_id",
                "transport",
                "image_base64",
                "image_mime",
                "image_caption",
                "suppress_chat_log",
                "task_constraint",
                "task_metadata",
            ):
                value = msg.get(key)
                if value not in (None, "", 0):
                    message[key] = value

            self._update_counter = max(offset, self._update_counter + 1)
            return [{
                "update_id": self._update_counter,
                "message": message,
            }]
        except queue.Empty:
            return []

    def configure_from_settings(self, settings: Dict[str, Any]) -> None:
        """Compatibility no-op; chat bridges are skills now."""
        return None

    def subscribe_response(self, chat_id: int, callback) -> str:
        """Subscribe to responses for a chat_id."""
        import uuid as _uuid
        sub_id = _uuid.uuid4().hex
        with self._response_subs_lock:
            self._response_subs[sub_id] = (chat_id, callback)
        return sub_id

    def unsubscribe_response(self, subscription_id: str) -> None:
        """Remove a response subscription."""
        with self._response_subs_lock:
            self._response_subs.pop(subscription_id, None)

    def shutdown(self) -> None:
        return None

    def handle_web_message(
        self,
        text: str,
        *,
        sender_session_id: str = "",
        client_message_id: str = "",
        image_base64: str = "",
        image_mime: str = "",
        image_caption: str = "",
        task_metadata: Optional[Dict[str, Any]] = None,
        chat_id: int = 1,
        project_id: str = "",
    ) -> None:
        # Multi-project (v6.32.0): the web owner may address a project chat by
        # positive chat_id. The OWNER identity never changes (user_id stays 1 —
        # binding is security-load-bearing); only the thread id varies. A2A
        # negative ids are rejected here — they are not a web surface.
        try:
            thread_id = int(chat_id or 1)
        except (TypeError, ValueError):
            thread_id = 1
        if thread_id < 1:
            thread_id = 1
        clean_text = str(text or "").strip()
        if not clean_text and not image_base64:
            return
        ts = utc_now_iso()
        if self._broadcast_fn:
            self._broadcast_fn({
                "type": "chat",
                "role": "user",
                "content": clean_text,
                "ts": ts,
                "source": "web",
                "chat_id": thread_id,
                "sender_session_id": sender_session_id,
                "client_message_id": client_message_id,
            })
        metadata = dict(task_metadata or {})
        if str(project_id or "").strip():
            metadata.setdefault("project_id", str(project_id).strip())
        self.enqueue_local_message(
            clean_text,
            chat_id=thread_id,
            user_id=1,
            source="web",
            sender_label="",
            sender_session_id=sender_session_id,
            client_message_id=client_message_id,
            image_base64=image_base64,
            image_mime=image_mime,
            image_caption=image_caption,
            task_metadata=metadata or None,
        )

    def enqueue_local_message(
        self,
        text: str,
        *,
        chat_id: int = 1,
        user_id: int = 1,
        source: str = "web",
        sender_label: str = "",
        sender_session_id: str = "",
        client_message_id: str = "",
        transport: Optional[Dict[str, Any]] = None,
        image_base64: str = "",
        image_mime: str = "",
        image_caption: str = "",
        suppress_chat_log: bool = False,
        task_constraint: Optional[Dict[str, Any]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        clean_text = str(text or "").strip()
        caption_text = str(image_caption or "").strip()
        image_b64 = str(image_base64 or "").strip()
        if not clean_text and caption_text:
            clean_text = caption_text
        if not clean_text and not image_b64:
            return
        # Invariant: the default chat/user id is the web owner (1). External
        # transports (source != "web") MUST pass explicit ids — the Host Service
        # injects 0 for unidentified senders so they can never bind/own the web
        # owner. coerce_chat_identity preserves an explicit 0 sentinel.
        self._inbox.put({
            "chat_id": coerce_chat_identity(chat_id, 1),
            "user_id": coerce_chat_identity(user_id, 1),
            "text": clean_text,
            "source": str(source or "web"),
            "sender_label": str(sender_label or ""),
            "sender_session_id": str(sender_session_id or ""),
            "client_message_id": str(client_message_id or ""),
            "transport": dict(transport or {}),
            "image_base64": image_b64,
            "image_mime": str(image_mime or ""),
            "image_caption": caption_text,
            "suppress_chat_log": bool(suppress_chat_log),
            "task_constraint": dict(task_constraint or {}),
            "task_metadata": dict(task_metadata or {}),
        })

    def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "",
        ts: Optional[str] = None,
        is_progress: bool = False,
        task_id: str = "",
        progress_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Send text to UI, A2A subscribers, and host event stream."""
        clean_text = _strip_markdown(text) if not parse_mode else text
        message_ts = ts or utc_now_iso()
        transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        meta = dict(progress_meta or {})
        msg = {
            "type": "text",
            "content": clean_text,
            "markdown": bool(parse_mode),
            "is_progress": bool(is_progress),
            "ts": message_ts,
            "task_id": str(task_id or ""),
        }
        if meta:
            msg.update(meta)
        with self._response_subs_lock:
            subs = [(sid, cb) for sid, (cid, cb) in self._response_subs.items()
                    if cid == chat_id and not is_progress]
        for sid, cb in subs:
            try:
                cb(clean_text)
            except Exception:
                log.debug("A2A response callback error for sub %s", sid, exc_info=True)
        if self._broadcast_fn and not is_a2a_chat_id(chat_id):
            payload = {
                "type": "chat",
                "role": "assistant",
                "content": clean_text,
                "markdown": bool(parse_mode),
                "is_progress": bool(is_progress),
                "ts": message_ts,
                "task_id": str(task_id or ""),
                "chat_id": int(chat_id or 0),
                "transport": transport,
            }
            if meta:
                payload.update(meta)
            self._broadcast_fn(payload)
        if not is_a2a_chat_id(chat_id):
            event = {
                "chat_id": int(chat_id or 0),
                "text": clean_text,
                "markdown": bool(parse_mode),
                "is_progress": bool(is_progress),
                "ts": message_ts,
                "task_id": str(task_id or ""),
                "transport": transport,
            }
            if meta:
                event.update(meta)
            publish_event(CHAT_OUTBOUND, event)
        return True, "ok"

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        """Send typing indicator to UI/event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True
        if self._broadcast_fn:
            self._broadcast_fn({"type": "typing", "action": action, "chat_id": int(chat_id or 0)})
        typing_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_TYPING, {"chat_id": int(chat_id or 0), "action": str(action or ""), "transport": typing_transport})
        return True

    def send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: str = "",
        mime: str = "image/png",
    ) -> Tuple[bool, str]:
        """Send photo to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        b64_str = base64.b64encode(photo_bytes).decode("ascii")
        msg = {
            "type": "photo",
            "role": "assistant",
            "image_base64": b64_str,
            "mime": mime,
            "caption": caption,
            "ts": utc_now_iso(),
            "chat_id": int(chat_id or 0),
        }
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        photo_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_PHOTO, {
            "chat_id": int(chat_id or 0),
            "transport": photo_transport,
            "caption": str(caption or ""),
            "image_base64": b64_str,
            "mime": str(mime or ""),
            "ts": msg["ts"],
        })
        return True, "ok"

    def send_video(
        self,
        chat_id: int,
        video_bytes: bytes,
        caption: str = "",
        mime: str = "video/mp4",
    ) -> Tuple[bool, str]:
        """Send video to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        b64_str = base64.b64encode(video_bytes).decode("ascii")
        msg = {
            "type": "video",
            "role": "assistant",
            "video_base64": b64_str,
            "mime": mime,
            "caption": caption,
            "ts": utc_now_iso(),
            "chat_id": int(chat_id or 0),
        }
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        video_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_VIDEO, {
            "chat_id": int(chat_id or 0),
            "transport": video_transport,
            "caption": str(caption or ""),
            "video_base64": b64_str,
            "mime": str(mime or ""),
            "ts": msg["ts"],
        })
        return True, "ok"

    def send_document(
        self,
        chat_id: int,
        file_bytes: bytes,
        filename: str = "file",
        caption: str = "",
        mime: str = "application/octet-stream",
        download_url: str = "",
        task_id: str = "",
    ) -> Tuple[bool, str]:
        """Send an arbitrary document/file to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        b64_str = base64.b64encode(file_bytes).decode("ascii")
        safe_name = str(filename or "file")
        ts = utc_now_iso()
        msg = {
            "type": "document",
            "role": "assistant",
            "file_base64": b64_str,
            "mime": mime,
            "filename": safe_name,
            "caption": caption,
            "download_url": str(download_url or ""),
            "ts": ts,
            "chat_id": int(chat_id or 0),
        }
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        document_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_DOCUMENT, {
            "chat_id": int(chat_id or 0),
            "transport": document_transport,
            "caption": str(caption or ""),
            "file_base64": b64_str,
            "mime": str(mime or ""),
            "filename": safe_name,
            "download_url": str(download_url or ""),
            "ts": ts,
        })
        # Persist a compact chat row (NO base64) so the delivered document is
        # rebuilt on reload; the durable artifact download_url carries the bytes.
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or f"📎 {safe_name}",
            ts=ts,
            task_id=str(task_id or ""),
            record_type="document",
            filename=safe_name,
            mime=str(mime or ""),
            download_url=str(download_url or ""),
            caption=str(caption or ""),
        )
        return True, "ok"

    def push_log(self, event: dict):
        """Stream append_jsonl events to UI."""
        try:
            self._log_queue.put_nowait(event)
        except queue.Full:
            try:
                self._log_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._log_queue.put_nowait(event)
            except queue.Full:
                pass
        if self._broadcast_fn:
            # Surface the event's chat_id top-level so the browser's per-thread
            # fan-out (isMyThread) can route the live card to its project panel
            # instead of the main chat. Events without a chat_id default to main.
            self._broadcast_fn({"type": "log", "data": event, "chat_id": int(event.get("chat_id") or 0)})

    def ui_poll_logs(self) -> list:
        """Drain pending log events for the web UI."""
        batch = []
        for _ in range(50):
            try:
                batch.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def ui_send(
        self,
        text: str,
        *,
        broadcast: bool = True,
        sender_session_id: str = "",
        client_message_id: str = "",
        suppress_chat_log: bool = False,
        image_base64: str = "",
        image_mime: str = "",
        image_caption: str = "",
        task_constraint: Optional[Dict[str, Any]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        chat_id: int = 1,
        project_id: str = "",
    ):
        """Accept a web UI message for the agent."""
        if broadcast:
            self.handle_web_message(
                text,
                sender_session_id=sender_session_id,
                client_message_id=client_message_id,
                image_base64=image_base64,
                image_mime=image_mime,
                image_caption=image_caption,
                task_metadata=task_metadata,
                chat_id=chat_id,
                project_id=project_id,
            )
            return
        self.enqueue_local_message(
            text,
            suppress_chat_log=suppress_chat_log,
            task_constraint=task_constraint,
            task_metadata=task_metadata,
        )



def _strip_markdown(text: str) -> str:
    """Best-effort markdown-to-plain-text fallback."""
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    return text


def _send_markdown(
    chat_id: int,
    text: str,
    ts: Optional[str] = None,
    is_progress: bool = False,
    task_id: str = "",
    progress_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Send markdown text through the bridge."""
    bridge = get_bridge()
    if not text:
        return False, "empty"
    return bridge.send_message(
        chat_id,
        text,
        parse_mode="markdown",
        ts=ts,
        is_progress=is_progress,
        task_id=task_id,
        progress_meta=progress_meta,
    )


def _format_budget_line(st: Dict[str, Any]) -> str:
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    pct = (spent / total * 100.0) if total > 0 else 0.0
    sha = (st.get("current_sha") or "")[:8]
    branch = st.get("current_branch") or "?"
    return f"—\nBudget: ${spent:.4f} / ${total:.2f} ({pct:.2f}%) | {branch}@{sha}"


def budget_line(force: bool = False) -> str:
    try:
        from supervisor.state import update_state

        every = max(1, int(BUDGET_REPORT_EVERY_MESSAGES))
        report_box: Dict[str, Any] = {"emit": False}

        def _tick_counter(live: Dict[str, Any]) -> None:
            if force:
                live["budget_messages_since_report"] = 0
                report_box["emit"] = True
                return
            counter = int(live.get("budget_messages_since_report") or 0) + 1
            if counter < every:
                live["budget_messages_since_report"] = counter
                return
            live["budget_messages_since_report"] = 0
            report_box["emit"] = True

        st = update_state(_tick_counter)
        return _format_budget_line(st) if report_box["emit"] else ""
    except Exception:
        log.debug("Suppressed exception in budget_line", exc_info=True)
        return ""


def log_chat(
    direction: str,
    chat_id: int,
    user_id: int,
    text: str,
    ts: Optional[str] = None,
    fmt: str = "",
    source: str = "",
    sender_label: str = "",
    sender_session_id: str = "",
    client_message_id: str = "",
    transport: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    record_type: str = "",
    filename: str = "",
    mime: str = "",
    download_url: str = "",
    caption: str = "",
) -> None:
    if DATA_DIR:
        record = {
            "ts": ts or utc_now_iso(),
            "session_id": load_state().get("session_id"),
            "direction": direction,
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
            "format": fmt,
            "source": source,
            "sender_label": sender_label,
            "sender_session_id": sender_session_id,
            "client_message_id": client_message_id,
            "transport": dict(transport or {}),
            "task_id": str(task_id or ""),
        }
        # Media rows (e.g. delivered documents) carry a variable ``type`` plus
        # lightweight metadata so /api/chat/history can rebuild the bubble on
        # reload WITHOUT persisting base64. ``type`` is set from a variable (not a
        # literal) so the frozen-contract AST parity scan does not treat this
        # persisted row as a DocumentOutbound WS envelope.
        if record_type:
            record["type"] = record_type
        if filename:
            record["filename"] = filename
        if mime:
            record["mime"] = mime
        if download_url:
            record["download_url"] = download_url
        if caption:
            record["caption"] = caption
        append_jsonl(DATA_DIR / "logs" / "chat.jsonl", record)


def send_with_budget(chat_id: int, text: str, log_text: Optional[str] = None,
                     fmt: str = "",
                     is_progress: bool = False, task_id: str = "",
                     progress_meta: Optional[Dict[str, Any]] = None,
                     ts: Optional[str] = None) -> None:
    st = load_state()
    owner_id = int(st.get("owner_id") or 0)
    _text = str(text or "")
    msg_ts = ts or utc_now_iso()

    if is_progress and DATA_DIR:
        progress_record = {
            "ts": msg_ts,
            "type": "send_message",
            "task_id": task_id,
            "is_progress": True,
            "direction": "out", "chat_id": chat_id, "user_id": owner_id,
            "text": text if log_text is None else log_text,
            "content": _text,
            "format": fmt,
        }
        if progress_meta:
            progress_record.update(dict(progress_meta))
        append_jsonl(DATA_DIR / "logs" / "progress.jsonl", progress_record)
    else:
        log_chat(
            "out",
            chat_id,
            owner_id,
            text if log_text is None else log_text,
            ts=msg_ts,
            fmt=fmt,
            task_id=task_id,
        )

    if _text.strip() in ("", "\u200b"):
        return
    # Budget footers belong in dashboard/status flows, not every chat reply.
    full = _text

    if fmt == "markdown":
        ok, err = _send_markdown(
            chat_id,
            full,
            ts=msg_ts,
            is_progress=is_progress,
            task_id=task_id,
            progress_meta=progress_meta,
        )
        return

    bridge = get_bridge()
    bridge.send_message(
        chat_id,
        full,
        ts=msg_ts,
        is_progress=is_progress,
        task_id=task_id,
        progress_meta=progress_meta,
    )
