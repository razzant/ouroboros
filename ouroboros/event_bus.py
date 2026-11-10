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
CHAT_QUIZ_STATE = "chat.quiz_state"
CHAT_CARD = "chat.card"
SKILL_LIFECYCLE = "skill.lifecycle"
VALID_TOPICS = frozenset({CHAT_OUTBOUND, CHAT_TYPING, CHAT_PHOTO, CHAT_VIDEO, CHAT_DOCUMENT, CHAT_LINKS,
                          CHAT_QUIZ, CHAT_QUIZ_STATE, CHAT_CARD, SKILL_LIFECYCLE})


@dataclass
class EventSubscription:
    id: str
    skill_name: str
    topic: str
    handler: Callable[[Dict[str, Any]], Any]


class EventBus:
    """Subscriptions plus the one host loop supervised tasks and async subscribers share.

    That loop is a daemon thread (``extension-loop``), not the HTTP loop.
    ``set_loop`` still binds a caller-owned loop for tests and does not stop it.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._loop_gate = threading.RLock()
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._owns_loop = False

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Bind a loop this bus does not own. An owned extension loop is stopped first."""
        with self._loop_gate:
            if loop is self._loop and self._owns_loop:
                return
            if self._owns_loop:
                self.stop_loop()
            self._loop = loop
            self._loop_thread = None
            self._owns_loop = False

    def start_loop(self) -> asyncio.AbstractEventLoop:
        """Start ``extension-loop`` and return only once that loop is running.

        A supervised task scheduled in the next boot step sees ``is_running``;
        otherwise it is dropped and never starts.
        """
        with self._loop_gate:
            if self._owns_loop and self._loop is not None and self._loop.is_running():
                return self._loop
            if self._owns_loop:
                self.stop_loop()
            ready = threading.Event()
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(target=_run, name="extension-loop", daemon=True)
            thread.start()
            # A thread that never reaches run_forever is a broken runtime, not a judgment.
            if not ready.wait(5.0):
                loop.call_soon_threadsafe(loop.stop)
                thread.join(5.0)
                if not thread.is_alive():
                    loop.close()
                raise RuntimeError("extension loop did not start")
            self._loop = loop
            self._loop_thread = thread
            self._owns_loop = True
            return loop

    def stop_loop(self, timeout: float = 5.0) -> None:
        """Cancel every task on the owned extension loop, drain, stop and close it.

        ``timeout`` bounds the drain and the thread join so a call blocked inside
        the loop cannot hold process exit. A loop installed with ``set_loop`` is
        left running. A task's ``finally`` runs when cancellation reaches it;
        a call that never returns to the loop outlives this bound.
        """
        with self._loop_gate:
            if not self._owns_loop or self._loop is None:
                return
            loop, thread = self._loop, self._loop_thread
            self._loop = None
            self._loop_thread = None
            self._owns_loop = False
        _shutdown_extension_loop(loop, thread, timeout)

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
            except Exception as exc:
                from ouroboros.deployment_extensions import discloses_failure, failure_words

                log.debug("Event subscriber %s failed for topic %s: %s", sub.id, topic,
                          failure_words(sub.skill_name, exc), exc_info=discloses_failure(sub.skill_name))

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            return {
                sub_id: {"skill_name": sub.skill_name, "topic": sub.topic}
                for sub_id, sub in self._subscriptions.items()
            }


def _shutdown_extension_loop(
    loop: asyncio.AbstractEventLoop, thread: threading.Thread | None, timeout: float,
) -> None:
    if loop.is_running():
        async def _drain() -> None:
            current = asyncio.current_task()
            while True:
                pending = [task for task in asyncio.all_tasks(loop) if task is not current and not task.done()]
                if not pending:
                    return
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_drain(), loop).result(timeout)
        except Exception:
            log.debug("extension loop drain did not finish within %.1fs", timeout, exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout)
    if thread is None or not thread.is_alive():
        try:
            loop.close()
        except Exception:
            log.debug("extension loop close failed", exc_info=True)


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
    "CHAT_QUIZ_STATE",
    "CHAT_TYPING",
    "CHAT_VIDEO",
    "EventBus",
    "SKILL_LIFECYCLE",
    "VALID_TOPICS",
    "get_global_event_bus",
    "init_global_event_bus",
    "publish_event",
]
