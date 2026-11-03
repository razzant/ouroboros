"""Keep Telegram's native typing indicator alive while a reply is being prepared.

One ``sendChatAction`` lives about five seconds on Telegram clients (or until
the bot's next message), while preparing a reply can take minutes (#1766).
``TypingCue`` is the bridge's ONE refresher, a supervised task. What it shows is
the host's current state (``ouroboros.gateway.reply_preparation``, DESIGN §9):
roots whose reply the owner awaits, whose own work or a working helper is
active, and whose answer has not been handed off — mapped through the bridge's
own mirror policy (``_target_chat``) to Telegram chats. ``chat.typing`` and
``chat.outbound`` only wake it; it never infers a task's life from an event.

- Idle: no state read at all until a wake. The first read at start rediscovers
  quiet preparation after a skill reload.
- Active: one fresh read per wake or renewal (wakes inside ``_MIN_READ_GAP_SEC``
  share one), and per Telegram chat at most one action every ``_REFRESH_SEC``
  however many roots, helpers or events back it; a message that reached the chat
  cleared its indicator, so the next read may renew it at once.
- A read that fails or verifies no positive sends nothing; while the last
  verified state was active it is retried a bounded number of times, then the
  cue idles.
- Ordering: while a message goes to a chat no action starts there, a read begun
  before it sends nothing there, and an action already in flight gets up to
  ``_ORDER_WAIT_SEC`` to land first; past that it is cancelled locally. A request
  Telegram already received may still be accepted after the message, and the
  client may show an accepted action for up to five seconds: neither is undone.

Fail-soft, from Timik232's original refresher: transient failures (transport,
429, 5xx) back off per chat on the bridge's shared pacing and recover on the
first success; a permanent rejection mutes that chat with one warning until a
message reaches it again. No ``asyncio.wait_for``: on Python 3.10 it can swallow
a concurrent cancel. Unloading the skill cancels the task and its in-flight
sends; a late wake is then a no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Optional, Set

from .telegram_api import (
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramRequestRejected,
    is_transient_telegram_error,
    next_telegram_retry_delay,
)

log = logging.getLogger(__name__)

_REFRESH_SEC = 3.5  # per-chat renewal spacing, below the ~5 s action lifetime
_MIN_READ_GAP_SEC = 1.0  # wakes inside this gap share one read
_ORDER_WAIT_SEC = 2.0  # how long a message waits for an in-flight action to its chat
_READ_RETRIES = 4  # unverified reads retried after a verified active state


def host_reply_preparation(data_dir: Any) -> Dict[str, Any]:
    """The host's projection; an older host without it is a read failure (no cue)."""
    from ouroboros.gateway.reply_preparation import reply_preparation

    return reply_preparation(data_dir)


class TypingCue:
    """One supervised refresher for every Telegram chat the bridge mirrors replies to."""

    def __init__(self, *, read_state: Callable[[], Dict[str, Any]],
                 targets: Callable[[Dict[str, Any]], Set[int]],
                 client_factory: Callable[[], Any], warn: Callable[[str], None]) -> None:
        self._read_state, self._targets = read_state, targets
        self._client_factory, self._warn = client_factory, warn
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: Optional[asyncio.Event] = None
        self._inflight: Dict[int, "asyncio.Future[None]"] = {}
        self._fence: Dict[int, int] = {}  # per chat: messages begun, the read fence
        self._sending: Dict[int, int] = {}  # per chat: messages still on their way
        self._renewed: Dict[int, float] = {}  # per chat: when the last action started
        self._retry_at: Dict[int, float] = {}
        self._delay: Dict[int, float] = {}
        self._muted: Set[int] = set()
        self._last_read = float("-inf")

    def wake(self) -> None:
        """Something may have changed: re-read soon. No I/O; a no-op unless running."""
        loop, event = self._loop, self._event
        if loop is None or event is None or loop.is_closed():
            return
        try:
            same_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            same_loop = False
        if same_loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    @contextlib.asynccontextmanager
    async def message(self, chat_id: int) -> AsyncIterator[Callable[[], None]]:
        """Bracket one message to ``chat_id``; call the yielded function once it was delivered."""
        self._fence[chat_id] = self._fence.get(chat_id, 0) + 1
        self._sending[chat_id] = self._sending.get(chat_id, 0) + 1
        delivered = []
        try:
            sending = self._inflight.get(chat_id)
            if sending is not None and not sending.done():
                done, _ = await asyncio.wait({sending}, timeout=_ORDER_WAIT_SEC)
                if not done:
                    sending.cancel()  # local only: bytes Telegram already received stay received
            yield lambda: delivered.append(True)
        finally:
            self._sending[chat_id] -= 1
            if not self._sending[chat_id]:
                self._sending.pop(chat_id)
            if delivered:
                # The message cleared the chat's indicator, and a permanent typing rejection there is stale.
                self._muted.discard(chat_id)
                self._renewed.pop(chat_id, None)
            self.wake()

    async def run(self) -> None:
        """The supervised task body (restartable)."""
        self._loop, self._event = asyncio.get_running_loop(), asyncio.Event()
        self._event.set()  # the first read
        active, retries = False, 0
        try:
            while True:
                if not active:
                    await self._event.wait()
                gap = self._last_read + _MIN_READ_GAP_SEC - time.monotonic()
                if gap > 0:
                    await asyncio.sleep(gap)
                self._event.clear()
                fence, self._last_read = dict(self._fence), time.monotonic()
                try:
                    state = await asyncio.to_thread(self._read_state)
                    chats = self._targets(state) - self._muted  # a muted chat waits for a message, not a read
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.debug("Telegram typing state unavailable", exc_info=True)
                    state, chats = {"complete": False}, set()
                if chats:
                    active, retries = True, 0
                    await self._pause(self._renew(chats, fence))
                elif active and state.get("complete") is False and retries < _READ_RETRIES:
                    retries += 1
                    await self._pause(_REFRESH_SEC * 2 ** retries)
                else:
                    active, retries = False, 0
        finally:
            self._event = None
            for sending in self._inflight.values():
                sending.cancel()
            self._inflight.clear()

    async def _pause(self, seconds: float) -> None:
        """Sleep ``seconds`` or until a wake; ``asyncio.wait`` cancels nothing on its timeout."""
        assert self._event is not None
        waiter = asyncio.ensure_future(self._event.wait())
        try:
            await asyncio.wait({waiter}, timeout=seconds)
        finally:
            waiter.cancel()

    def _renew(self, chats: Set[int], fence: Dict[int, int]) -> float:
        """Start each due action; return how long until the next one falls due."""
        now, due = time.monotonic(), _REFRESH_SEC
        for chat in sorted(chats):
            if (self._fence.get(chat, 0) != fence.get(chat, 0) or chat in self._sending
                    or self._retry_at.get(chat, 0.0) > now):
                continue
            sending = self._inflight.get(chat)
            if sending is not None and not sending.done():
                continue
            wait = self._renewed.get(chat, float("-inf")) + _REFRESH_SEC - now
            if wait > 0:
                due = min(due, wait)
                continue
            self._renewed[chat] = now
            self._inflight[chat] = asyncio.ensure_future(self._send(chat))
        return max(due, _MIN_READ_GAP_SEC)

    async def _send(self, chat: int) -> None:
        try:
            await self._client_factory().send_chat_action(chat, "typing")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, TelegramRequestRejected) and not is_transient_telegram_error(exc):
                # Bad token, blocked bot, wrong chat: stop this chat once, loudly enough to be seen.
                self._muted.add(chat)
                self._warn(f"Telegram typing indicator stopped for this chat: {exc}")
                return
            previous = self._delay.get(chat)
            self._delay[chat] = next_telegram_retry_delay(previous) if previous else TELEGRAM_RETRY_INITIAL_SEC
            self._retry_at[chat] = time.monotonic() + self._delay[chat]
            log.debug("Telegram typing renewal deferred: %s", exc)
        else:
            self._delay.pop(chat, None)
            self._retry_at.pop(chat, None)
