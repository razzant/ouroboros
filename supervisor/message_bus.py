"""Queue-based bridge between UI/skill transports and the supervisor.

Chat, media, typing and log frames never carry A2A synthetic chats to the browser; ``project_thread`` is stamped from Project registry membership (not a numeric range), so external transport ids stay unstamped.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from ouroboros.artifacts import store_chat_media_bytes
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.event_bus import CHAT_DOCUMENT, CHAT_LINKS, CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ, CHAT_QUIZ_STATE, CHAT_TYPING, CHAT_VIDEO, publish_event
from ouroboros import dialogue_store
from ouroboros.store import StoreError, acquisition_budget
from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
from supervisor.state import load_state
from supervisor.message_ingress import (  # noqa: F401 - the named ingress's public names stay importable here
    _INGRESS_LOCK, accept_chat, accept_local_message, accepted_chat_message, log_chat, record_inbound_message,
    recover_pending_dispatches)
from ouroboros.project_dialogue import ROOM_FIXED_ROW_TYPES
from ouroboros.projects_registry import RoomResolver, stamp_project_thread
from ouroboros.tools.core import (
    LinkActionsValidationError,
    QuizValidationError,
    validate_link_actions,
    validate_quiz_payload,
)
from ouroboros.utils import utc_now_iso
from ouroboros.subagent_messages import is_task_card_message

log = logging.getLogger(__name__)


DATA_DIR = None  # pathlib.Path
TOTAL_BUDGET_LIMIT: float = 0.0
BUDGET_REPORT_EVERY_MESSAGES: int = 10
_BRIDGE: Optional["LocalChatBridge"] = None


class MessageTransportError(RuntimeError):
    """The physical send failed after its durable dialogue append."""


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


def row_chat_identity(*candidates: Any, default: int = 0) -> int:
    """The row's OWN address: the FIRST PRESENT candidate, coerced.

    Sibling of ``notification_chat_route`` for the other question. Routing asks
    "where does this notice go" and therefore suppresses synthetic A2A ids;
    a RECORD must not, or a synthetic row would be filed under an audience it
    never had. Addressing is honest — the broadcast choke is what keeps A2A
    traffic out of a human stream, not a rewritten record.
    """
    for candidate in candidates:
        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            continue
        return coerce_chat_identity(candidate, default)
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
    if drive_root is not None:
        try:  # the dispatches a crash left owed: handed over once, through the bus that opens now
            recover_pending_dispatches(chat_bridge, drive_root)
        except StoreError:
            log.warning("Owed dispatches could not be recovered at bus init", exc_info=True)


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
        from ouroboros.server_control import PanicIngress

        self.panic = PanicIngress()  # server binds the existing emergency owner
        self._inbox = queue.Queue()   # user -> agent
        # Updates the consumer handed back unprocessed (``requeue_updates``); memory only.
        self._replay: List[Dict[str, Any]] = []
        self._replay_lock = threading.Lock()
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
        """Return the waiting message and a bounded snapshot of its queued peers.

        Only the first read may block; the fixed qsize snapshot keeps a busy
        producer from holding the supervisor here, and each update keeps its own
        monotonic id and transport. Updates handed back through ``requeue_updates``
        come first, in order with their original ids. An item that cannot become
        an update is logged by source identity (an accepted source as lost) and
        skipped alone, never discarding its dequeued or still-queued peers.
        """
        with self._replay_lock:
            if self._replay:
                replay, self._replay = self._replay, []
                return replay
        try:
            first = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return []
        updates: List[Dict[str, Any]] = []
        for index in range(1 + self._inbox.qsize()):
            try:
                raw_msg = first if index == 0 else self._inbox.get_nowait()
            except queue.Empty:
                break
            try:
                updates.append(self._build_update(raw_msg, offset))
            except Exception:
                # Do not log private text or attachment bytes. An accepted
                # delivery can still be located by its ingress coordinates.
                identity = raw_msg if isinstance(raw_msg, dict) else {}
                log.exception(
                    "Dropping one undecodable queued chat message; lost source=%s chat_id=%s client_message_id=%s; "
                    "%d earlier update(s) in this batch are kept",
                    str(identity.get("source") or "unknown")[:80],
                    str(identity.get("chat_id") or "unknown")[:80],
                    str(identity.get("client_message_id") or "unknown")[:120], len(updates),
                )
        return updates

    def _build_update(self, raw_msg: Any, offset: int) -> Dict[str, Any]:
        """Turn one dequeued item into an update with its own monotonic id."""
        msg = ({"chat_id": 1, "user_id": 1, "text": raw_msg, "source": "web", "sender_label": ""}
               if isinstance(raw_msg, str) else dict(raw_msg or {}))

        msg_chat_id = coerce_chat_identity(msg.get("chat_id"), 1)
        msg_user_id = coerce_chat_identity(msg.get("user_id"), 1)
        message = {
            "chat": {"id": msg_chat_id},
            "from": {"id": msg_user_id},
            "text": str(msg.get("text") or ""),
            "source": str(msg.get("source") or "web"),
        }
        self.activate_update_transport(msg)
        for key in (
            "sender_label", "sender_session_id", "client_message_id", "transport",
            "image_base64", "image_mime", "image_caption", "suppress_chat_log",
            "task_constraint", "task_metadata", "accepted_source_ref", "accepted_source_row", "received_at",
        ):
            value = msg.get(key)
            if value not in (None, "", 0):
                message[key] = value

        self._update_counter = max(offset, self._update_counter + 1)
        return {
            "update_id": self._update_counter,
            "message": message,
        }

    def requeue_updates(self, updates: List[Dict[str, Any]]) -> int:
        """Hand back dequeued updates the consumer did not process; returns how many were kept.

        This process's next ``get_updates`` serves them ahead of newer arrivals,
        order and ids intact, so a survived handler crash or a cut-short batch
        does not erase the messages behind it. Memory only: nothing outlives the
        hard exit of /panic or a completed /restart.
        """
        tail = [u for u in list(updates or []) if isinstance(u, dict) and u.get("message")]
        if not tail:
            return 0
        with self._replay_lock:
            self._replay.extend(tail)  # an earlier hand-back is older: it is served first
        return len(tail)

    def activate_update_transport(self, msg: Dict[str, Any]) -> None:
        """Bind this message's reply route (also at build time, for single-update readers).

        The supervisor rebinds per message: a later same-chat message in one
        drained batch must not steal an earlier one's reply transport.
        """
        chat_id = coerce_chat_identity(msg.get("chat_id", (msg.get("chat") or {}).get("id")), 1)
        transport = msg.get("transport")
        if isinstance(transport, dict) and transport and chat_id != 1:
            self._chat_transports[chat_id] = dict(transport)
        else:
            self._chat_transports.pop(chat_id, None)

    def chat_transport(self, chat_id: Any) -> Dict[str, Any]:
        """A copy of the reply transport bound to ``chat_id`` (``{}`` when none), as its events carry it."""
        return dict(self._chat_transports.get(int(chat_id or 0), {}) or {})

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
        self.panic.invalidate_owner()

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
        dispatch=None,
    ) -> None:
        from supervisor.message_ingress import handle_web_message
        return handle_web_message(self, text, sender_session_id=sender_session_id,
            client_message_id=client_message_id, image_base64=image_base64, image_mime=image_mime,
            image_caption=image_caption, task_metadata=task_metadata, chat_id=chat_id,
            project_id=project_id, dispatch=dispatch)

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
        accepted_source_ref: Optional[Dict[str, Any]] = None,
        accepted_source_row: Optional[Dict[str, Any]] = None,
        received_at: str = "",
    ) -> None:
        """The ONE ingress every transport enqueues through, so it stamps the host receipt time
        ``received_at`` for all of them: an earlier host stamp of this message (its accepted row's
        time, the WS acceptance's ``client_surface.received_at``) is kept, else now. The update
        carries it; a surface fact without one (a host channel stamp) gets it too, so every
        record that copies the fact carries it. Panic uses the server's independent
        emergency owner; ordinary ingress is a dict put, no lock or I/O."""
        clean_text = str(text or "").strip()
        caption_text = str(image_caption or "").strip()
        image_b64 = str(image_base64 or "").strip()
        if not clean_text and caption_text:
            clean_text = caption_text
        if self.panic.request(clean_text, source=source, user_id=user_id, chat_id=chat_id):
            return
        if not clean_text and not image_b64 and not (task_metadata or {}).get("chat_attachment_uploads"):
            return  # nothing to say and nothing attached (a file-only message carries uploads)
        surface = (task_metadata or {}).get("client_surface")
        surface = surface if isinstance(surface, dict) and surface else None
        received_at = str(received_at or (surface or {}).get("received_at") or "") or utc_now_iso()
        if surface is not None and not surface.get("received_at"):
            task_metadata = {**(task_metadata or {}), "client_surface": {**surface, "received_at": received_at}}
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
            "accepted_source_ref": dict(accepted_source_ref or {}),
            "accepted_source_row": dict(accepted_source_row or {}),
            "received_at": received_at,
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
        transport = self.chat_transport(chat_id)
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
        project_id: str = "",
        project_chat_id: int = 0,
        status: str = "accepted",
        options: Optional[List[Dict[str, Any]]] = None,
        attachment_manifest: Optional[List[Dict[str, Any]]] = None,
        routing_token: str = "",
        cause: str = "", reasoning_effort: str = "",
        project_name: str = "", target_task_ids=None, target_titles=None,
    ) -> None:
        """Emit a typed routing receipt without creating an assistant bubble.

        Web consumes the WS annotation; non-Web skills receive the same additive typed envelope on
        their existing outbound subscription.  ``text`` stays empty and ``suppress_bubble`` is
        explicit, so legacy transports do not invent a human-visible mirror message.
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
        if target_task_ids is not None:
            payload.update(project_name=project_name, target_task_ids=list(target_task_ids), target_titles=list(target_titles or []))
        if str(target_label or ""):
            payload["target_label"] = str(target_label)
        if project_id and int(project_chat_id) > 0:
            payload.update(project_id=str(project_id), project_chat_id=int(project_chat_id))
        if str(routing_token or ""):
            # #198: the picker card's click identity; presentation-only frames without it stay text lines.
            payload["routing_token"] = str(routing_token)
        if str(cause or ""):
            # Q3=A: the host-owned owner-facing sentence for a refused act.
            payload["cause"] = str(cause)
        if str(reasoning_effort or ""):  # #1539: the start a New task picked from this card requests
            payload["reasoning_effort"] = str(reasoning_effort)
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
                "transport": self.chat_transport(chat_id),
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

        ``kind`` is stamped only for registry-tracked direct turns
        (``direct_chat``); RUNNING queue roots are stamped ``managed_task`` at
        the event handler, and children and untracked tasks stay empty. The
        stamp is kept for wire compatibility only (no in-repo client reads it)
        and grants nothing: the web header never admits a typing frame into its
        live-activity set, into which only the /api/state census inserts. Telegram's
        native typing consumer ignores ``kind`` entirely.
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
        publish_event(CHAT_TYPING, {
            "chat_id": int(chat_id or 0),
            "action": str(action or ""),
            "activity_id": str(activity_id or ""),
            "client_message_id": str(client_message_id or ""),
            "phase": str(phase or "thinking"),
            "kind": str(kind or ""),
            "transport": self.chat_transport(chat_id),
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
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        stored = log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or "Photo attachment",
            ts=msg["ts"],
            task_id=str(task_id or ""),
            record_type="photo", feed=True,
            mime=str(mime or ""),
            download_url=download_url,
            download_url_compat=download_url_compat,
            caption=str(caption or ""),
        )
        if stored:
            from ouroboros.chat_references import projection_fields
            msg.update(projection_fields(stored))
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        publish_event(CHAT_PHOTO, {
            "chat_id": int(chat_id or 0),
            "transport": self.chat_transport(chat_id),
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
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        stored = log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or "Video attachment",
            ts=msg["ts"],
            task_id=str(task_id or ""),
            record_type="video", feed=True,
            mime=str(mime or ""),
            download_url=download_url,
            download_url_compat=download_url_compat,
            caption=str(caption or ""),
        )
        if stored:
            from ouroboros.chat_references import projection_fields
            msg.update(projection_fields(stored))
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        publish_event(CHAT_VIDEO, {
            "chat_id": int(chat_id or 0),
            "transport": self.chat_transport(chat_id),
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
        file_ref: Optional[Dict[str, Any]] = None,
        download_url_compat: str = "",
    ) -> Tuple[bool, str]:
        """Send an arbitrary document/file to UI and host event subscribers."""
        if is_a2a_chat_id(chat_id):
            return True, "ok"
        if file_ref is not None:
            from ouroboros.gateway.files import resolve_task_file_reference
            try:
                resolve_task_file_reference(DATA_DIR, task_id or "interactive", file_ref)
                if file_bytes:
                    from hashlib import sha256
                    if len(file_bytes) != file_ref["size"] or sha256(file_bytes).hexdigest() != file_ref["sha256"]:
                        raise ValueError("inline document bytes disagree with the captured file")
            except (OSError, ValueError, TypeError) as exc:
                return False, f"Document source unavailable: {exc}"
            download_url = f"/api/tasks/{quote(task_id or 'interactive', safe='')}/artifacts/{quote(str(file_ref['path']), safe='')}"
        size_bytes = file_ref["size"] if file_ref is not None else len(file_bytes)
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
            "download_url_compat": str(download_url_compat or ""),
            "file_ref": dict(file_ref or {}),
            "size_bytes": size_bytes,
            "ts": ts,
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        stored = log_chat(
            "out",
            int(chat_id or 0),
            owner_id,
            caption or f"📎 {safe_name}",
            ts=ts,
            task_id=str(task_id or ""),
            record_type="document", feed=True,
            filename=safe_name,
            mime=str(mime or ""),
            download_url=str(download_url or ""),
            caption=str(caption or ""),
            size_bytes=size_bytes,
            download_url_compat=download_url_compat,
        )
        if stored:
            from ouroboros.chat_references import projection_fields
            msg.update(projection_fields(stored))
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        publish_event(CHAT_DOCUMENT, {
            "chat_id": int(chat_id or 0),
            "transport": self.chat_transport(chat_id),
            "caption": str(caption or ""),
            "file_base64": b64_str,
            "mime": str(mime or ""),
            "filename": safe_name,
            "download_url": str(download_url or ""),
            "download_url_compat": str(download_url_compat or ""),
            "file_ref": dict(file_ref or {}),
            "ts": ts,
        })
        # Persist a compact chat row (NO base64) so the delivered document is
        # rebuilt on reload; the durable artifact download_url carries the bytes.
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
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        stored = log_chat(
            "out", int(chat_id or 0), owner_id, safe_title, ts=ts,
            task_id=str(task_id or ""), record_type="links", feed=True,
            actions=validated, title=safe_title,
        )
        if stored:
            from ouroboros.chat_references import projection_fields
            msg.update(projection_fields(stored))
        stamp_project_thread(DATA_DIR, msg)
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        publish_event(CHAT_LINKS, {
            "chat_id": int(chat_id or 0),
            "transport": self.chat_transport(chat_id),
            "title": safe_title,
            "actions": validated,
            "ts": ts,
        })
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
        wait_for_answer: bool = False,
        host_facts: str = "",
    ) -> Tuple[bool, str]:
        """Send an owner quiz card to the UI and host event subscribers.

        ``host_facts`` (the host's sentence under the question) rides the frame, the
        event and the chat row when non-empty; ``project_name`` rides the EVENT only."""
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
        from ouroboros.owner_quiz import CLOSURE_FIELDS, question_delivery_projection, quiz_states, record_publication
        from supervisor.log_addressing import resolve_quiz_publication

        prior = quiz_states(DATA_DIR, str(task_id)).get(qid) or {}
        # Validation concerns the original ask, not its still-active wait. Both
        # timeout and terminal reconciliation retain wait_ended_at as evidence
        # that this required question was valid without an optional assumption.
        required_ask = bool(prior.get("wait_for_answer") or prior.get("wait_ended_at")) if prior else wait_for_answer
        try:
            payload = validate_quiz_payload(question, options, stake, assumption, wait_for_answer=required_ask)
        except QuizValidationError as exc:
            return False, str(exc)
        chat_id = resolve_quiz_publication(DATA_DIR, str(task_id), prior, chat_id)
        block = record_publication(DATA_DIR, str(task_id), qid, chat_id)
        if not block or "publication_chat_id" not in block:
            return False, "canonical quiz publication could not be persisted"
        chat_id = int(block["publication_chat_id"])
        state = str(block.get("state") or state)
        # A queued ask may arrive after answer, close or resumed waiting. Render
        # the same canonical card the decision ingress now validates.
        details = block.get("option_details") or []
        payload = {
            "question": str(block.get("question") or ""),
            "options": [{"label": str(label),
                         **({"detail": str(details[index])} if index < len(details) and details[index] else {}),
                         **({"recommended": True} if block.get("recommended_index") == index else {})}
                        for index, label in enumerate(block.get("options") or [])],
            "stake": str(block.get("stake") or ""), "assumption": str(block.get("assumption") or ""),
        }
        wait_for_answer = bool(block.get("wait_for_answer"))
        host_facts = str(block.get("host_facts") or host_facts or "")
        lifecycle = {key: block[key] for key in ("answered_index", "comment", "wait_ended_at", *CLOSURE_FIELDS) if key in block}
        if delivery := question_delivery_projection(block):
            lifecycle["question_delivery"] = delivery
        materials = list(block.get("materials") or [])
        if block.get("presence_contact"):
            lifecycle["presence_contact"] = block["presence_contact"]
        ts = utc_now_iso()
        msg = {
            "type": "quiz",
            "role": "assistant",
            "quiz_id": qid,
            "question": payload["question"],
            "wait_for_answer": bool(wait_for_answer),
            "options": payload["options"],
            "stake": payload["stake"],
            "assumption": payload["assumption"],
            "state": str(state or "open"),
            "ts": ts,
            "chat_id": int(chat_id or 0),
            "task_id": str(task_id or ""),
        }
        msg.update(lifecycle)
        if materials:
            msg["materials"] = materials
        if host_facts:
            msg["host_facts"] = str(host_facts)  # the envelope literal keeps constant keys (contract scan)
        stamp_project_thread(DATA_DIR, msg)
        project = None
        if msg.get("project_thread"):
            try:
                from ouroboros.projects_registry import list_reserved_projects

                project = next((row for row in list_reserved_projects(DATA_DIR)
                                if row.get("chat_id") == int(chat_id)), None)
            except Exception:
                log.debug("Quiz project lookup failed", exc_info=True)
        try:
            owner_id = int(load_state().get("owner_id") or 0)
        except Exception:
            owner_id = 0
        stored = log_chat(
            "out", int(chat_id or 0), owner_id, payload["question"], ts=ts,
            task_id=str(task_id or ""), record_type="quiz", feed=True,
            client_message_id=f"quiz_publication:{task_id}:{qid}",
            require_write=True,
            quiz={
                "quiz_id": qid,
                **lifecycle,
                **({"materials": materials} if materials else {}),
                "wait_for_answer": bool(wait_for_answer),
                **{key: payload[key] for key in ("options", "stake", "assumption")},
                "state": str(state or "open"),
                **({"host_facts": str(host_facts)} if host_facts else {}),
            },
        )
        if stored:
            from ouroboros.chat_references import projection_fields
            msg.update(projection_fields(stored))
            from ouroboros.owner_quiz import mark_published
            mark_published(DATA_DIR, str(task_id), qid)
            # The durable ask now exists. Expose its settled delivery fact, not
            # the pre-append snapshot carried by the history write above.
            if delivery := question_delivery_projection(quiz_states(DATA_DIR, str(task_id)).get(qid)):
                lifecycle["question_delivery"] = delivery
                msg["question_delivery"] = delivery
        if self._broadcast_fn:
            self._broadcast_fn(msg)
        quiz_transport = self.chat_transport(chat_id)
        publish_event(CHAT_QUIZ, {
            "chat_id": int(chat_id or 0),
            "transport": quiz_transport,
            "wait_for_answer": bool(wait_for_answer),
            "quiz_id": qid,
            "task_id": str(task_id or ""),
            "question": payload["question"],
            "options": payload["options"],
            "stake": payload["stake"],
            "assumption": payload["assumption"],
            "state": str(state or "open"),
            "ts": ts,
            **lifecycle,
            **({"materials": materials} if materials else {}),
            **({"host_facts": str(host_facts)} if host_facts else {}),
            **({"project_name": str(project["name"])} if project and project.get("name") else {}),
        })
        if self._broadcast_fn and msg.get("project_thread"):
            try:
                from ouroboros.owner_quiz import quiz_states
                from ouroboros.project_dialogue import project_question_pointer

                pointer = project_question_pointer(msg, quiz_states(DATA_DIR, task_id).get(qid), project)
                if pointer:
                    frame = {
                        "type": "chat", "role": pointer["role"], "content": pointer["text"],
                        "ts": pointer["ts"], "system_type": pointer["system_type"],
                        "task_id": pointer["task_id"], "quiz_id": pointer["quiz_id"],
                        "quiz_state": pointer["quiz_state"], "project_id": pointer["project_id"],
                        "project_name": pointer["project_name"], "project_chat_id": pointer["project_chat_id"],
                        "chat_id": pointer["chat_id"], "is_progress": False, "markdown": False,
                        "owner_wait_state": pointer.get("owner_wait_state", ""),
                        "source_status": pointer.get("source_status", ""),
                    }
                    # The complete pointer row (ChatOutbound mirrors): present only when known.
                    for key in ("question", "options", "option_details", "stake", "assumption", "recommended_index",
                                "answered_index", "comment", "wait_for_answer", "wait_ended_at",
                                "owner_wait_resume_reason", "host_facts", "materials", "closed_reason", "question_delivery"):
                        if key in pointer:
                            frame[key] = pointer[key]
                    frame.update(projection_fields(stored))
                    self._broadcast_fn(frame)
            except Exception:
                # The question is already delivered. History/activity reads heal
                # this derived view without another quiz or paid execution.
                log.debug("Project question pointer broadcast failed", exc_info=True)
        return True, "ok"

    def send_quiz_state(
        self,
        quiz_id: str,
        task_id: str,
        state: str,
        answered_index: Optional[int] = None,
        chat_id: int = 0,
        comment: Optional[str] = None,
        wait_for_answer: Optional[bool] = None,
        closed_reason: Optional[str] = None,
        closed_at: Optional[str] = None,
        closed_actor: Optional[str] = None,
        closed_request_id: Optional[str] = None,
        closed_by_task_id: Optional[str] = None,
        question_delivery: Optional[Dict[str, str]] = None,
    ) -> None:
        """Broadcast a quiz lifecycle update to already-rendered cards.

        A SEPARATE WS discriminator ("quiz_state", contracts mirror): the
        display path dedupes "quiz" frames by quiz_id+ts, so a state change
        must never masquerade as a new card. Durability lives in the
        owner_quiz task-result projection (history replay merges it) — this
        frame is the live half only, so a lost broadcast heals on reload.
        ``comment`` is the owner's recorded free-text answer (#471): the live
        card renders it exactly as the replayed one does; absent when empty.
        """
        msg: Dict[str, Any] = {
            "type": "quiz_state",
            "quiz_id": str(quiz_id or ""),
            "task_id": str(task_id or ""),
            "state": str(state or ""),
            "ts": utc_now_iso(),
        }
        if closed_reason:
            msg["closed_reason"] = str(closed_reason)
        if closed_at:
            msg["closed_at"] = str(closed_at)
        if closed_actor:
            msg["closed_actor"] = str(closed_actor)
        if closed_request_id:
            msg["closed_request_id"] = str(closed_request_id)
        if closed_by_task_id:
            msg["closed_by_task_id"] = str(closed_by_task_id)
        if question_delivery:
            msg["question_delivery"] = dict(question_delivery)
        if answered_index is not None:
            msg["answered_index"] = int(answered_index)
        if str(comment or ""):
            msg["comment"] = str(comment)
        if wait_for_answer is not None:
            # Additive: ``False`` after a bounded wait closed — the card stops saying
            # "waiting" while it stays answerable.
            msg["wait_for_answer"] = bool(wait_for_answer)
        if int(chat_id or 0):
            msg["chat_id"] = int(chat_id or 0)
        if self._broadcast_fn:
            try:
                self._broadcast_fn(msg)
            except Exception:
                log.debug("quiz_state broadcast failed", exc_info=True)
        # The WebSocket reaches only the SPA. Transport skills need the same
        # lifecycle fact to edit an already delivered card without a reload.
        quiz_transport = self.chat_transport(chat_id)
        publish_event(CHAT_QUIZ_STATE, {**msg, "transport": quiz_transport})

    def push_log(self, event: dict):
        """Stream append_jsonl events to UI."""
        if self._broadcast_fn and not is_a2a_chat_id(event.get("chat_id")):
            # Task-scoped events arrive addressed (supervisor/log_addressing.py); an unaddressable one keeps the chat-0 default (Main admits it).
            frame = {"type": "log", "data": event, "chat_id": int(event.get("chat_id") or 0)}
            stamp_project_thread(DATA_DIR, frame)
            self._broadcast_fn(frame)

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
        dispatch=None,
    ):
        """Accept a web UI message for the agent."""
        if broadcast:
            self.handle_web_message(text, sender_session_id=sender_session_id, client_message_id=client_message_id,
                                    image_base64=image_base64, image_mime=image_mime, image_caption=image_caption,
                                    task_metadata=task_metadata, chat_id=chat_id, project_id=project_id,
                                    dispatch=dispatch)
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

        from supervisor.state import StateUnavailable

        try:
            st = update_state(_tick_counter)
        except StateUnavailable:  # a budget report is bookkeeping; the message itself proceeds
            return ""
        if not report_box["emit"]:
            return ""
        display_state = dict(st)
        try:
            if DATA_DIR is None:
                raise RuntimeError("message bus data root is not initialized")
            from ouroboros.usage_accounting import usage_projection, usage_writer_snapshot

            total = float(TOTAL_BUDGET_LIMIT or 0.0)
            accounting = (  # display of scalars, sent from the supervisor loop too: no per-root map
                usage_projection(DATA_DIR, global_limit_usd=total, include_roots=False, allow_stale=True)
                if total > 0
                else usage_writer_snapshot(DATA_DIR, allow_stale=True)
            )
            display_state["spent_usd"] = float(accounting.get("settled_usd") or 0.0)  # known; holds in the detail
            display_state["usage_accounting"] = accounting
            display_state["_budget_accounting_available"] = True
        except Exception:
            log.error("Physical-attempt ledger unavailable for budget line", exc_info=True)
            display_state["_budget_accounting_available"] = False
        return _format_budget_line(display_state)
    except Exception:
        log.debug("Suppressed exception in budget_line", exc_info=True)
        return ""


def send_with_budget(chat_id: int, text: str, log_text: Optional[str] = None,
                     fmt: str = "",
                     is_progress: bool = False, task_id: str = "",
                     progress_meta: Optional[Dict[str, Any]] = None,
                     ts: Optional[str] = None,
                     role: str = "", system_type: str = "",
                     narration: Optional[bool] = None,
                     require_write: bool = False,
                     ensure_record_boundary: bool = False) -> Dict[str, Any]:
    """Send an owner-visible row with one voice and timestamp for history and live.

    Narration True means model speech, False host speech; absent legacy rows keep
    their old reading. Normalize new host sends once so startup/restart/evolution,
    command replies and provider incidents do not become model card titles (#1011).
    Explicit narration (argument or progress_meta) wins. A named delivery is owed
    in the existing outbox and also requires a positive transport receipt. A
    required notice without one has one owner, its producer, which retries or
    records a failed write: it keeps the caller's wait and raises when unwritten.
    Return the selected destination so delivery receipts and mailbox handoffs
    retain this send's room even if task placement changes around the send.
    """
    producer_owned = require_write and not (progress_meta or {}).get("delivery_id")
    with (contextlib.nullcontext() if producer_owned
          else acquisition_budget(0 if is_progress and not require_write else ACQUISITION_SLICE_SEC)):
        st = load_state()
        owner_id = int(st.get("owner_id") or 0)
        _text = str(text or "")
        msg_ts = ts or utc_now_iso()
        meta = dict(progress_meta or {})
        if narration is not None:
            meta["narration"] = bool(narration)
        meta.setdefault("narration", False)
        from ouroboros.chat_references import origin_reply, projection_fields
        if not is_progress and (reply := origin_reply(DATA_DIR, task_id, meta if meta.get("origin_message_ref") else None)):
            meta.setdefault("reply_to", reply)
        progress_meta = meta
        if task_id and DATA_DIR and system_type not in ROOM_FIXED_ROW_TYPES and meta.get("_is_direct_chat") is not True:
            # The one room rule (A4) at the one send seam: a task-addressed host row
            # lands in its task's Project room on the durable row, the WebSocket frame
            # and every transport alike, whichever producer addressed it. A child row
            # that reached here without its lineage borrows it from the same cached
            # facts (never a result read), so its card nests it live and on replay.
            try:
                rooms = RoomResolver(DATA_DIR)
                chat_id = rooms(task_id, meta.get("parent_task_id"), meta.get("root_task_id")) or chat_id
                lineage = {} if meta.get("parent_task_id") else rooms.lineage(task_id)
                for key in ("parent_task_id", "root_task_id", "delegation_role"):
                    if lineage.get(key) and not meta.get(key):
                        meta[key] = lineage[key]
            except Exception:
                log.debug("room rule failed for %s; the producer's address stands", task_id, exc_info=True)

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
                "role": str(role or "") or "assistant",
                "system_type": str(system_type or ""),
            }
            if progress_meta:
                progress_record.update(dict(progress_meta))
            writer = dialogue_store.append if require_write else dialogue_store.append_quietly
            if writer(DATA_DIR, "progress", progress_record, idempotent=require_write) is not None:
                from ouroboros.task_activity_facts import record_narration
                record_narration(DATA_DIR, progress_record)
                meta.update(projection_fields(progress_record))
        else:
            feed = _text.strip() not in ("", "\u200b") and not is_task_card_message(progress_meta)
            stored = log_chat(
                # S3 (Q4): a typed SYSTEM row persists as direction="system", the
                # role history replay already maps to a system rendering — so the
                # receipt survives reload as system, never as Ouroboros's speech.
                "system" if role == "system" else "out",
                chat_id,
                owner_id,
                text if log_text is None else log_text,
                ts=msg_ts,
                fmt=fmt,
                source=str(meta.get("source") or ""),
                task_id=task_id,
                record_type=system_type,
                message_meta=progress_meta,
                require_write=require_write,
                ensure_record_boundary=ensure_record_boundary,
                feed=feed,
            )

        if not is_progress and stored:
            meta.update(projection_fields(stored))
        if _text.strip() in ("", "\u200b"):
            return {"chat_id": chat_id}
        # Budget footers belong in dashboard/status flows, not every chat reply.
        full = _text

        transport = dict(ts=msg_ts, is_progress=is_progress, task_id=task_id, progress_meta=progress_meta)
        try:
            if fmt == "markdown":
                receipt = _send_markdown(chat_id, full, **transport, role=role, system_type=system_type)
            else:
                receipt = get_bridge().send_message(chat_id, full, **transport, role=role, system_type=system_type)
            if meta.get("delivery_id"):
                ok, err = receipt
                if not ok:
                    raise MessageTransportError(f"message transport refused: {err}")
        except Exception as exc:
            raise MessageTransportError(str(exc)) from exc
        return {"chat_id": chat_id}
