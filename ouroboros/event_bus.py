"""Typed in-process event bus for reviewed extension subscriptions."""

from __future__ import annotations

import asyncio
import inspect
import logging
import pathlib
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

CHAT_OUTBOUND = "chat.outbound"
CHAT_TYPING = "chat.typing"
CHAT_PHOTO = "chat.photo"
CHAT_VIDEO = "chat.video"
CHAT_DOCUMENT = "chat.document"
CHAT_LINKS = "chat.links"
CHAT_QUIZ = "chat.quiz"
CHAT_QUIZ_STATE = "chat.quiz_state"
SKILL_LIFECYCLE = "skill.lifecycle"
# One owner-notification fact ("come back, there is something here for you"):
# a durable `owner_notification` row in logs/events.jsonl — which the server
# log sink already turns into ONE live `log` frame for browsers — plus this
# topic for skill subscribers (the Telegram skill mirrors it to the phone).
# Never a chat row, never model context.
OWNER_NOTIFICATION = "owner.notification"
VALID_TOPICS = frozenset({
    CHAT_OUTBOUND, CHAT_TYPING, CHAT_PHOTO, CHAT_VIDEO, CHAT_DOCUMENT, CHAT_LINKS, CHAT_QUIZ, CHAT_QUIZ_STATE,
    SKILL_LIFECYCLE, OWNER_NOTIFICATION,
})
OWNER_NOTIFICATION_TEXT_CHARS = 1000
OWNER_NOTIFICATION_KEY_CHARS = 128
_OWNER_NOTIFICATION_CATEGORY_CHARS = 32


@dataclass
class EventSubscription:
    id: str
    skill_name: str
    topic: str
    handler: Callable[[Dict[str, Any]], Any]


class EventBus:
    def __init__(self):
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def subscribe(
        self,
        skill_name: str,
        topic: str,
        handler: Callable[[Dict[str, Any]], Any],
        *,
        sub_id: Optional[str] = None,
    ) -> str:
        """Attach one subscription; ``sub_id`` lets a staged registration
        pre-mint the id it already returned to the caller (ABI-9: the bus
        attach is deferred to publication, the id is not)."""
        if topic not in VALID_TOPICS:
            raise ValueError(f"unsupported event topic: {topic}")
        sid = str(sub_id) if sub_id else uuid.uuid4().hex
        with self._lock:
            self._subscriptions[sid] = EventSubscription(
                id=sid,
                skill_name=str(skill_name or ""),
                topic=topic,
                handler=handler,
            )
        return sid

    def unsubscribe(self, sub_id: str) -> None:
        with self._lock:
            self._subscriptions.pop(str(sub_id or ""), None)

    def unsubscribe_skill(self, skill_name: str) -> None:
        with self._lock:
            for sub_id, sub in list(self._subscriptions.items()):
                if sub.skill_name == skill_name:
                    self._subscriptions.pop(sub_id, None)

    def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """Invoke every live subscription of *topic* with a copy of *data*.

        Copy semantics (ABI-9 unload residual, by design): the subscriber
        list is COPIED under the bus lock and the handlers run with NO lock
        held, so ``unsubscribe`` guarantees only that a publish STARTING
        after it returns will not deliver; a publish that already copied the
        handler may still invoke it after the unsubscribe. The extension
        unload path therefore unsubscribes (and closes the runtime API)
        BEFORE removing surfaces, and a late in-flight delivery is a no-op
        against the host.
        """
        if topic not in VALID_TOPICS:
            raise ValueError(f"unsupported event topic: {topic}")
        with self._lock:
            subscribers = [sub for sub in self._subscriptions.values() if sub.topic == topic]
        payload = dict(data or {})
        payload.setdefault("topic", topic)
        for sub in subscribers:
            try:
                result = sub.handler(payload)
                if inspect.isawaitable(result):
                    if self._loop is not None and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(result, self._loop)
                    else:
                        log.debug("Dropping async event handler without running event loop for %s", topic)
            except Exception:
                log.debug("Event subscriber %s failed for topic %s", sub.id, topic, exc_info=True)

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            return {
                sub_id: {"skill_name": sub.skill_name, "topic": sub.topic}
                for sub_id, sub in self._subscriptions.items()
            }


_GLOBAL_BUS: Optional[EventBus] = None


def init_global_event_bus() -> EventBus:
    global _GLOBAL_BUS
    _GLOBAL_BUS = EventBus()
    return _GLOBAL_BUS


def get_global_event_bus() -> EventBus:
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = EventBus()
    return _GLOBAL_BUS


def publish_event(topic: str, data: Dict[str, Any]) -> None:
    get_global_event_bus().publish(topic, data)


def owner_notification_chat_id(drive_root) -> int:
    """Where an owner notification goes: the owner's chat of THIS data root
    (its own state file, never the process-global one), else Main while no
    owner is bound. Deliberately not ``notification_chat_route``: chat 0 is a
    real destination there (the Skill Review panel), but a banner addressed to
    it reaches nobody — the browser notifier refuses it. One rule for every
    producer (the Host route, the scheduler, a future agent tool)."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.utils import read_json_dict

    state = read_json_dict(pathlib.Path(drive_root) / "state" / "state.json") or {}
    try:
        owner = int(state.get("owner_chat_id") or 0)
    except (TypeError, ValueError):
        owner = 0
    return owner if owner > 0 else WEB_UI_CHAT_ID


def emit_owner_notification(
    drive_root: Any, *, chat_id: int, category: str, text: str, source: str,
    key: str = "", scheduled_for: str = "", publish: bool = True,
) -> Optional[Dict[str, Any]]:
    """Emit one owner notification: the durable row, the live frame, the topic.

    Returns the row when the durable append landed (the live browser frame is
    that append's log-sink copy, so it exists exactly when the row does) and
    ``None`` when it did not — the caller words its status honestly. The topic
    publish is best-effort after the row; ``publish=False`` leaves it to the
    caller (the scheduler tick appends under its table lock and publishes the
    collected rows after releasing it, so a slow subscriber never holds the
    supervisor).

    SERVER PROCESS ONLY: the bus is process-local, so a worker-side call would
    persist the row (and reach browsers through the worker sink) while the
    topic publish silently reaches no subscriber. A producer inside a task must
    hand the fact to the supervisor instead (the skill.lifecycle event queue is
    the precedent). ``chat_id`` must be a positive owner-visible chat: the
    browser notifier refuses the hidden partition and A2A ids, and a row
    without ``task_id`` never joins any task's model context.
    """
    from ouroboros.utils import append_jsonl, utc_now_iso

    body = str(text or "").strip()
    kind = str(category or "").strip()
    origin = str(source or "").strip()
    if not body or len(body) > OWNER_NOTIFICATION_TEXT_CHARS:
        raise ValueError(f"notification text must be 1..{OWNER_NOTIFICATION_TEXT_CHARS} characters")
    if not kind or len(kind) > _OWNER_NOTIFICATION_CATEGORY_CHARS:
        raise ValueError("notification category is required")
    if not origin:
        raise ValueError("notification source is required")
    dedupe = str(key or "").strip()
    if len(dedupe) > OWNER_NOTIFICATION_KEY_CHARS:
        raise ValueError(f"notification key must be at most {OWNER_NOTIFICATION_KEY_CHARS} characters")
    if type(chat_id) is not int or chat_id <= 0:
        raise ValueError("notification chat_id must be a positive owner chat id")
    row: Dict[str, Any] = {
        "ts": utc_now_iso(), "type": "owner_notification", "category": kind,
        "text": body, "source": origin, "key": dedupe, "chat_id": int(chat_id),
    }
    if str(scheduled_for or "").strip():
        row["scheduled_for"] = str(scheduled_for).strip()
    try:
        # This row is a receipt (the scheduler consumes its schedule on it), so
        # it repairs a torn predecessor's boundary before appending and reports
        # an unwritable log as "not written" instead of raising into the seam.
        written = append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", dict(row),
                               ensure_record_boundary=True)
    except OSError:
        log.warning("owner notification could not be appended", exc_info=True)
        return None
    if not written:
        return None
    if publish:
        publish_owner_notification(row)
    return row


def publish_owner_notification(row: Dict[str, Any]) -> None:
    """Best-effort topic publish of an already-persisted notification row."""
    try:
        publish_event(OWNER_NOTIFICATION, dict(row))
    except Exception:
        log.debug("owner notification topic publish failed", exc_info=True)


__all__ = [
    "OWNER_NOTIFICATION_KEY_CHARS",
    "owner_notification_chat_id",
    "CHAT_DOCUMENT",
    "CHAT_LINKS",
    "CHAT_OUTBOUND",
    "CHAT_PHOTO",
    "CHAT_QUIZ",
    "CHAT_QUIZ_STATE",
    "CHAT_TYPING",
    "CHAT_VIDEO",
    "EventBus",
    "OWNER_NOTIFICATION",
    "OWNER_NOTIFICATION_TEXT_CHARS",
    "SKILL_LIFECYCLE",
    "VALID_TOPICS",
    "emit_owner_notification",
    "get_global_event_bus",
    "init_global_event_bus",
    "publish_event",
    "publish_owner_notification",
]
