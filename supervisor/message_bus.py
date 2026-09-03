"""Queue-based bridge between UI/skill transports and the supervisor."""

from __future__ import annotations

import base64
import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from ouroboros.artifacts import store_chat_media_bytes
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.event_bus import CHAT_DOCUMENT, CHAT_LINKS, CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ, CHAT_TYPING, CHAT_VIDEO, publish_event
from supervisor.state import append_jsonl, load_state
from ouroboros.projects_registry import stamp_project_thread
from ouroboros.tools.core import (
    LinkActionsValidationError,
    QuizValidationError,
    validate_link_actions,
    validate_quiz_payload,
)
from ouroboros.utils import utc_now_iso
from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS

log = logging.getLogger(__name__)


DATA_DIR = None  # pathlib.Path
TOTAL_BUDGET_LIMIT: float = 0.0
BUDGET_REPORT_EVERY_MESSAGES: int = 10
_BRIDGE: Optional["LocalChatBridge"] = None


def _chat_media_download_url(task_id: str, data: bytes, mime: str) -> Tuple[str, str]:
    """Return ``(canonical_url, compat_url)`` for stored outbound chat media.

    The canonical form is the durable task-artifact route. The compat form is
    the long-shipped ``/api/files/download`` route, emitted only when the stored
    file actually resolves inside the current file-browser root. Packaged
    desktop launchers gate their file bridge to a fixed URL allowlist that
    predates the artifact route, so on an older launcher the canonical URL is
    refused while the compat URL still opens; it is a second address for the
    same bytes, never a replacement.
    """
    if not DATA_DIR:
        return "", ""
    try:
        stored = store_chat_media_bytes(DATA_DIR, task_id, data, mime)
    except Exception:
        log.warning("Could not persist outbound chat media", exc_info=True)
        return "", ""
    if not stored:
        return "", ""
    canonical = f"/api/tasks/{quote(task_id, safe='')}/artifacts/{quote(str(stored['name']), safe='')}"
    compat = ""
    stored_path = str(stored.get("path") or "") if isinstance(stored, dict) else ""
    if stored_path:
        try:
            from ouroboros.gateway.files import download_url_for_local_file

            compat = download_url_for_local_file(stored_path)
        except Exception:  # non-fatal: the canonical URL still works everywhere
            compat = ""
    return canonical, compat


def coerce_chat_identity(value: Any, default: int = 1) -> int:
    """Preserve explicit 0 sentinels while defaulting missing IDs for web chat."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def notification_chat_route(*candidates: Any) -> Optional[int]:
    """The first DELIVERABLE chat among ``candidates``, or None when there is none.

    ONE normalizer for every "where does this notice go" decision (C4), because
    the numeric id is not a boolean: **0 is the Skill Review panel** — a real
    destination — while **1 is Main** and a NEGATIVE id is synthetic A2A traffic
    that must never enter a human stream (`chat_id_policy`). Every producer that
    tested `if chat_id:` therefore did two wrong things at once: it dropped a
    panel-bound notice entirely, and it re-routed panel work to the owner chat.

    Membership decides: an ABSENT candidate (None / "" / unparseable) and a
    SUPPRESSED one (negative) both fall through to the next candidate, so a
    caller can express "the task's own chat, else the owner chat" as the argument
    order. None comes back only when no candidate is deliverable.
    """
    for candidate in candidates:
        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if is_a2a_chat_id(value):
            continue
        return value
    return None


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


def _advance_project_visible_revision(chat_id: int) -> None:
    """Advance unread only for a real owner-visible Project presentation row."""
    if DATA_DIR is None:
        return
    try:
        from ouroboros.projects_registry import increment_project_visible_revision

        increment_project_visible_revision(DATA_DIR, chat_id=int(chat_id or 0))
    except Exception:
        log.debug("Project visible-revision update failed for chat %s", chat_id, exc_info=True)


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
            echo = {
                "type": "chat",
                "role": "user",
                "content": clean_text,
                "ts": ts,
                "source": "web",
                "chat_id": thread_id,
                "sender_session_id": sender_session_id,
                "client_message_id": client_message_id,
            }
            stamp_project_thread(DATA_DIR, echo)
            self._broadcast_fn(echo)
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
        role: str = "",
        system_type: str = "",
    ) -> Tuple[bool, str]:
        """Send text to UI, A2A subscribers, and host event stream.

        ``role``/``system_type`` (S3, additive): a host-authored SYSTEM receipt
        carries its typed role end to end — live WebSocket frame, CHAT_OUTBOUND
        skill event — so it can never be rendered as Ouroboros's own speech
        (Q4 non-mimicry). Absent = the historical assistant framing.
        """
        # Text rides VERBATIM end to end: the live frame carries the same
        # bytes as the durable chat.jsonl row, and plain-vs-rich presentation
        # is the client's decision via the ``markdown`` flag (system rows
        # without it render escaped). The old best-effort strip here predated
        # that client contract and only made live diverge from history replay.
        message_ts = ts or utc_now_iso()
        transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        meta = dict(progress_meta or {})
        msg = {
            "type": "text",
            "content": text,
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
                cb(text)
            except Exception:
                log.debug("A2A response callback error for sub %s", sid, exc_info=True)
        if self._broadcast_fn and not is_a2a_chat_id(chat_id):
            payload = {
                "type": "chat",
                "role": str(role or "") or "assistant",
                "content": text,
                "markdown": bool(parse_mode),
                "is_progress": bool(is_progress),
                "ts": message_ts,
                "task_id": str(task_id or ""),
                "chat_id": int(chat_id or 0),
                "transport": transport,
            }
            if system_type:
                payload["system_type"] = str(system_type)
            if meta:
                payload.update(meta)
            # Last writer on the FINAL chat_id: meta cannot spoof/erase the stamp.
            payload.pop("project_thread", None)
            stamp_project_thread(DATA_DIR, payload)
            self._broadcast_fn(payload)
        if not is_a2a_chat_id(chat_id):
            event = {
                "chat_id": int(chat_id or 0),
                "text": text,
                "markdown": bool(parse_mode),
                "is_progress": bool(is_progress),
                "ts": message_ts,
                "task_id": str(task_id or ""),
                "transport": transport,
            }
            if role:
                event["role"] = str(role)
            if system_type:
                event["system_type"] = str(system_type)
            if meta:
                event.update(meta)
            publish_event(CHAT_OUTBOUND, event)
        return True, "ok"

    def send_routing_ack(
        self,
        chat_id: int,
        *,
        client_message_id: str,
        action: str,
        target: str = "",
        target_label: str = "",
        status: str = "accepted",
        options: Optional[List[Dict[str, Any]]] = None,
        attachment_manifest: Optional[List[Dict[str, Any]]] = None,
        routing_token: str = "",
    ) -> None:
        """Emit a typed routing receipt without creating an assistant bubble.

        Web consumes the WS annotation; non-Web skills receive the same additive
        typed envelope on their existing outbound subscription.  ``text`` stays
        empty and ``suppress_bubble`` is explicit, so legacy transports do not
        invent a human-visible mirror message.
        """
        payload = {
            "type": "message_annotation",
            "annotation_type": "routing_ack",
            "chat_id": int(chat_id or 0),
            "client_message_id": str(client_message_id or ""),
            "action": str(action or ""),
            "target": str(target or ""),
            "status": str(status or "accepted"),
            "suppress_bubble": True,
            "ts": utc_now_iso(),
        }
        if str(target_label or ""):
            payload["target_label"] = str(target_label)
        if str(routing_token or ""):
            # #198: the picker card's click identity; presentation-only frames
            # without it stay text lines.
            payload["routing_token"] = str(routing_token)
        if options is not None:
            payload["options"] = [dict(row) for row in options if isinstance(row, dict)]
        if attachment_manifest is not None:
            payload["attachment_manifest"] = [
                dict(row) for row in attachment_manifest if isinstance(row, dict)
            ]
        if self._broadcast_fn and not is_a2a_chat_id(chat_id):
            self._broadcast_fn(payload)
        if not is_a2a_chat_id(chat_id):
            publish_event(CHAT_OUTBOUND, {
                **payload,
                "text": "",
                "transport": dict(self._chat_transports.get(int(chat_id or 0), {}) or {}),
            })
        if (
            str(status or "") == "needs_manual_target"
            and isinstance(options, list) and options
            and not is_a2a_chat_id(chat_id)
        ):
            # Owner decision 4=A (#198): the routing LLM must later ground a
            # plain "2" reply against EXACTLY the list the owner was shown, so
            # the rendered numbered list becomes a durable outbound history row
            # (type="routing_options"; web history skips it — the picker card
            # is its richer rendering there; Telegram renders its own copy).
            from ouroboros.project_dialogue import routing_option_label

            labels = [routing_option_label(row) for row in options]
            labels = [label for label in labels if label]
            if labels:
                # Mirror the push-transport rendering exactly (top 8 + tail):
                # a numbered Telegram reply grounds against THESE numbers.
                shown = labels[:8]
                lines = ["I couldn't pick a destination for the last message. Options:"]
                lines.extend(f"{index}. {label}" for index, label in enumerate(shown, 1))
                if len(labels) > len(shown):
                    lines.append(f"…and {len(labels) - len(shown)} more in the web chat.")
                log_chat(
                    "out", int(chat_id or 0), 0, "\n".join(lines),
                    source="routing_picker", record_type="routing_options",
                )

    def send_chat_action(
        self,
        chat_id: int,
        action: str = "typing",
        *,
        activity_id: str = "",
        client_message_id: str = "",
        phase: str = "thinking",
        kind: str = "",
    ) -> bool:
        """Send typing indicator to UI/event subscribers.

        ``kind`` is stamped only for registry-tracked direct/ephemeral turns
        (``direct_chat``/``ephemeral_decision``); queued managed tasks emit
        typing without it, so the client knows the /api/state snapshot has no
        deletion authority over their entries.
        """
        if is_a2a_chat_id(chat_id):
            return True
        payload: Dict[str, Any] = {
            "type": "typing",
            "action": action,
            "chat_id": int(chat_id or 0),
        }
        stamp_project_thread(DATA_DIR, payload)
        if activity_id:
            payload["activity_id"] = str(activity_id)
        if client_message_id:
            payload["client_message_id"] = str(client_message_id)
        if phase:
            payload["phase"] = str(phase)
        if kind:
            payload["kind"] = str(kind)

        if self._broadcast_fn:
            self._broadcast_fn(payload)
        typing_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_TYPING, {
            "chat_id": int(chat_id or 0),
            "action": str(action or ""),
            "activity_id": str(activity_id or ""),
            "client_message_id": str(client_message_id or ""),
            "phase": str(phase or "thinking"),
            "kind": str(kind or ""),
            "transport": typing_transport,
        })
        return True

    def send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: str = "",
        mime: str = "image/png",
        task_id: str = "",
    ) -> Tuple[bool, str]:
        """Send photo to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        download_url, download_url_compat = _chat_media_download_url(task_id, photo_bytes, mime)
        b64_str = base64.b64encode(photo_bytes).decode("ascii")
        msg = {
            "type": "photo",
            "role": "assistant",
            "image_base64": b64_str,
            "mime": mime,
            "caption": caption,
            "ts": utc_now_iso(),
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        # The durable addresses ride the LIVE frame too: without them a
        # packaged desktop shell can only save this media after a history
        # reload (the bridge cannot be handed a data: URI).
        if download_url:
            msg["download_url"] = download_url
        if download_url_compat:
            msg["download_url_compat"] = download_url_compat
        stamp_project_thread(DATA_DIR, msg)
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
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or "Photo attachment",
            ts=msg["ts"],
            task_id=str(task_id or ""),
            record_type="photo",
            mime=str(mime or ""),
            download_url=download_url,
            download_url_compat=download_url_compat,
            caption=str(caption or ""),
        )
        _advance_project_visible_revision(chat_id)
        return True, "ok"

    def send_video(
        self,
        chat_id: int,
        video_bytes: bytes,
        caption: str = "",
        mime: str = "video/mp4",
        task_id: str = "",
    ) -> Tuple[bool, str]:
        """Send video to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        download_url, download_url_compat = _chat_media_download_url(task_id, video_bytes, mime)
        b64_str = base64.b64encode(video_bytes).decode("ascii")
        msg = {
            "type": "video",
            "role": "assistant",
            "video_base64": b64_str,
            "mime": mime,
            "caption": caption,
            "ts": utc_now_iso(),
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        # The durable addresses ride the LIVE frame too: without them a
        # packaged desktop shell can only save this media after a history
        # reload (the bridge cannot be handed a data: URI).
        if download_url:
            msg["download_url"] = download_url
        if download_url_compat:
            msg["download_url_compat"] = download_url_compat
        stamp_project_thread(DATA_DIR, msg)
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
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or "Video attachment",
            ts=msg["ts"],
            task_id=str(task_id or ""),
            record_type="video",
            mime=str(mime or ""),
            download_url=download_url,
            download_url_compat=download_url_compat,
            caption=str(caption or ""),
        )
        _advance_project_visible_revision(chat_id)
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
            "size_bytes": len(file_bytes),
            "ts": ts,
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        stamp_project_thread(DATA_DIR, msg)
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
            size_bytes=len(file_bytes),
        )
        _advance_project_visible_revision(chat_id)
        return True, "ok"

    def send_links(
        self,
        chat_id: int,
        actions: List[Dict[str, str]],
        title: str = "",
        task_id: str = "",
    ) -> Tuple[bool, str]:
        """Send validated HTTP(S) actions to UI and host event subscribers."""
        safe_title = str(title or "")[:240]
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        try:
            validated = validate_link_actions(actions)
        except LinkActionsValidationError as exc:
            return False, str(exc)
        ts = utc_now_iso()
        msg = {
            "type": "links",
            "role": "assistant",
            "title": safe_title,
            "actions": validated,
            "ts": ts,
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        links_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_LINKS, {
            "chat_id": int(chat_id or 0),
            "transport": links_transport,
            "title": safe_title,
            "actions": validated,
            "ts": ts,
        })
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        log_chat(
            "out", int(chat_id or 0), owner_id, safe_title, ts=ts,
            task_id=str(task_id or ""), record_type="links",
            actions=validated, title=safe_title,
        )
        _advance_project_visible_revision(chat_id)
        return True, "ok"

    def send_quiz(
        self,
        chat_id: int,
        quiz_id: str,
        question: str,
        options: List[Dict[str, str]],
        stake: str = "",
        assumption: str = "",
        state: str = "open",
        task_id: str = "",
    ) -> Tuple[bool, str]:
        """Send an owner quiz card to the UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        qid = str(quiz_id or "").strip()
        if not qid:
            return False, "quiz_id is required"
        if not str(task_id or "").strip():
            # The card's answer path is task-addressed (decision_id
            # "quiz:{task_id}:{quiz_id}"): an anonymous quiz would render
            # buttons that cannot deliver anywhere.
            return False, "task_id is required"
        try:
            payload = validate_quiz_payload(question, options, stake, assumption)
        except QuizValidationError as exc:
            return False, str(exc)
        ts = utc_now_iso()
        msg = {
            "type": "quiz",
            "role": "assistant",
            "quiz_id": qid,
            "question": payload["question"],
            "options": payload["options"],
            "stake": payload["stake"],
            "assumption": payload["assumption"],
            "state": str(state or "open"),
            "ts": ts,
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        quiz_transport = dict(self._chat_transports.get(int(chat_id or 0), {}) or {})
        publish_event(CHAT_QUIZ, {
            "chat_id": int(chat_id or 0),
            "transport": quiz_transport,
            "quiz_id": qid,
            "task_id": str(task_id or ""),
            "question": payload["question"],
            "options": payload["options"],
            "stake": payload["stake"],
            "assumption": payload["assumption"],
            "state": str(state or "open"),
            "ts": ts,
        })
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        log_chat(
            "out", int(chat_id or 0), owner_id, payload["question"], ts=ts,
            task_id=str(task_id or ""), record_type="quiz",
            quiz={
                "quiz_id": qid,
                "options": payload["options"],
                "stake": payload["stake"],
                "assumption": payload["assumption"],
                "state": str(state or "open"),
            },
        )
        _advance_project_visible_revision(chat_id)
        return True, "ok"

    def send_quiz_state(
        self,
        quiz_id: str,
        task_id: str,
        state: str,
        answered_index: Optional[int] = None,
        chat_id: int = 0,
    ) -> None:
        """Broadcast a quiz lifecycle update to already-rendered cards.

        A SEPARATE WS discriminator ("quiz_state", contracts mirror): the
        display path dedupes "quiz" frames by quiz_id+ts, so a state change
        must never masquerade as a new card. Durability lives in the
        owner_quiz task-result projection (history replay merges it) — this
        frame is the live half only, so a lost broadcast heals on reload.
        """
        if not self._broadcast_fn:
            return
        msg: Dict[str, Any] = {
            "type": "quiz_state",
            "quiz_id": str(quiz_id or ""),
            "task_id": str(task_id or ""),
            "state": str(state or ""),
            "ts": utc_now_iso(),
        }
        if answered_index is not None:
            msg["answered_index"] = int(answered_index)
        if int(chat_id or 0):
            msg["chat_id"] = int(chat_id or 0)
        try:
            self._broadcast_fn(msg)
        except Exception:
            log.debug("quiz_state broadcast failed", exc_info=True)

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
        if self._broadcast_fn and not is_a2a_chat_id(event.get("chat_id")):
            # Task-scoped events arrive already addressed
            # (supervisor/log_addressing.py); an unaddressable event keeps the
            # legacy chat-0 default, which Main still admits. A2A synthetic
            # chats are machine traffic and never enter the human stream.
            frame = {"type": "log", "data": event, "chat_id": int(event.get("chat_id") or 0)}
            stamp_project_thread(DATA_DIR, frame)
            self._broadcast_fn(frame)

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



def _send_markdown(
    chat_id: int,
    text: str,
    ts: Optional[str] = None,
    is_progress: bool = False,
    task_id: str = "",
    progress_meta: Optional[Dict[str, Any]] = None,
    role: str = "",
    system_type: str = "",
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
        role=role,
        system_type=system_type,
    )


def _format_budget_line(st: Dict[str, Any]) -> str:
    sha = (st.get("current_sha") or "")[:8]
    branch = st.get("current_branch") or "?"
    if st.get("_budget_accounting_available") is False:
        return f"—\nBudget: unavailable (physical-attempt ledger error) | {branch}@{sha}"
    spent = float(st.get("spent_usd") or 0.0)
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    pct = (spent / total * 100.0) if total > 0 else 0.0
    accounting = st.get("usage_accounting")
    if not isinstance(accounting, dict):
        accounting = {}
    confirmed = float(accounting.get("confirmed_usd") or 0.0)
    reserved = float(accounting.get("reserved_usd") or 0.0)
    unresolved = float(accounting.get("unresolved_upper_bound_usd") or 0.0)
    unknown = int(accounting.get("unknown_unmetered") or 0)
    cost_final = "yes" if accounting.get("cost_final") else "no"
    integrity = "DEGRADED (quarantined ledger tail)" if accounting.get("integrity_degraded") else "ok"
    detail = (
        f"confirmed ${confirmed:.4f}; reserved ${reserved:.4f}; "
        f"unresolved <=${unresolved:.4f}; unknown/unmetered {unknown}; "
        f"cost_final {cost_final}; ledger_integrity {integrity}"
    )
    return (
        f"—\nBudget: ${spent:.4f} / ${total:.2f} ({pct:.2f}%) "
        f"[{detail}] | {branch}@{sha}"
    )


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
        if not report_box["emit"]:
            return ""
        display_state = dict(st)
        try:
            if DATA_DIR is None:
                raise RuntimeError("message bus data root is not initialized")
            from ouroboros.usage_accounting import (
                ensure_legacy_imported,
                usage_breakdown,
                usage_projection,
            )
            from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC

            ensure_legacy_imported(DATA_DIR)
            total = float(TOTAL_BUDGET_LIMIT or 0.0)
            accounting = (
                usage_projection(
                    DATA_DIR,
                    global_limit_usd=total,
                    lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
                    allow_stale=True,
                )
                if total > 0
                else usage_breakdown(
                    DATA_DIR,
                    lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
                    allow_stale=True,
                )
            )
            display_state["spent_usd"] = float(accounting.get("accounted_usd") or 0.0)
            display_state["usage_accounting"] = accounting
            display_state["_budget_accounting_available"] = True
        except Exception:
            log.error("Physical-attempt ledger unavailable for budget line", exc_info=True)
            display_state["_budget_accounting_available"] = False
        return _format_budget_line(display_state)
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
    download_url_compat: str = "",
    caption: str = "",
    actions: Optional[List[Dict[str, str]]] = None,
    title: str = "",
    quiz: Optional[Dict[str, Any]] = None,
    size_bytes: Optional[int] = None,
    client_surface: Optional[Dict[str, Any]] = None,
    message_meta: Optional[Dict[str, Any]] = None,
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
        if client_surface:
            # Per-message sending-surface provenance (Owner Surface Fact);
            # optional column, never a "type":"chat" literal (AST-scan hygiene).
            record["client_surface"] = dict(client_surface)
        # Speech rows persist only the compact identity required to route a
        # child final after progress retention/rotation has removed its cards.
        meta = dict(message_meta or {})
        for key in SUBAGENT_MESSAGE_FIELDS:
            if key in meta:
                record[key] = meta[key]
        if record_type in ("project_started", "project_completion_summary"):
            for key in ("project_id", "project_name", "target_label", "status"):
                if key in meta:
                    record[key] = meta[key]
        if "task_terminal_status" in meta:
            record["task_terminal_status"] = str(meta.get("task_terminal_status") or "")
        if filename:
            record["filename"] = filename
        if mime:
            record["mime"] = mime
        if download_url:
            record["download_url"] = download_url
        if download_url_compat:
            # Second address for the same bytes, for desktop launchers whose
            # file bridge predates the task-artifact route.
            record["download_url_compat"] = download_url_compat
        if caption:
            record["caption"] = caption
        if actions:
            record["actions"] = [dict(action) for action in actions]
            record["title"] = str(title or "")
        if quiz:
            # Quiz rows persist the full card (no base64 anywhere) so history
            # replay rebuilds it with its lifecycle state.
            record["quiz"] = dict(quiz)
        if size_bytes is not None:
            record["size_bytes"] = int(size_bytes)
        append_jsonl(DATA_DIR / "logs" / "chat.jsonl", record)


def send_with_budget(chat_id: int, text: str, log_text: Optional[str] = None,
                     fmt: str = "",
                     is_progress: bool = False, task_id: str = "",
                     progress_meta: Optional[Dict[str, Any]] = None,
                     ts: Optional[str] = None,
                     role: str = "", system_type: str = "") -> None:
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
            # S3 (Q4): a typed SYSTEM row persists as direction="system", the
            # role history replay already maps to a system rendering — so the
            # receipt survives reload as system, never as Ouroboros's speech.
            "system" if role == "system" else "out",
            chat_id,
            owner_id,
            text if log_text is None else log_text,
            ts=msg_ts,
            fmt=fmt,
            task_id=task_id,
            record_type=system_type,
            message_meta=progress_meta,
        )

    if _text.strip() in ("", "\u200b"):
        return
    if (not is_progress) or bool((progress_meta or {}).get("task_incident")):
        _advance_project_visible_revision(chat_id)
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
            role=role,
            system_type=system_type,
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
        role=role,
        system_type=system_type,
    )
