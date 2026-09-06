"""Typed in-process event bus for reviewed extension subscriptions."""

from __future__ import annotations

import asyncio
import inspect
import logging
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
SKILL_LIFECYCLE = "skill.lifecycle"
VALID_TOPICS = frozenset({CHAT_OUTBOUND, CHAT_TYPING, CHAT_PHOTO, CHAT_VIDEO, CHAT_DOCUMENT, CHAT_LINKS, CHAT_QUIZ, SKILL_LIFECYCLE})


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


__all__ = [
    "CHAT_DOCUMENT",
    "CHAT_LINKS",
    "CHAT_OUTBOUND",
    "CHAT_PHOTO",
    "CHAT_QUIZ",
    "CHAT_TYPING",
    "CHAT_VIDEO",
    "EventBus",
    "SKILL_LIFECYCLE",
    "VALID_TOPICS",
    "get_global_event_bus",
    "init_global_event_bus",
    "publish_event",
]
