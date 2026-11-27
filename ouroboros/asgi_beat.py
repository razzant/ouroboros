"""The ASGI loop's own beat: how late the serving event loop runs, and where it stood when it fell behind.

The browser's liveness frame comes from ``server_runtime.ws_heartbeat_loop`` on the uvicorn event loop, the one
thread that also serves every request, socket and Panic frame. Each beat stamps ``ASGI_BEAT`` with its monotonic
time and its lateness (the interval it actually slept minus the one it asked for), and the loop thread's ident
once; each open socket keeps the largest lateness it lived through, and its disconnect line also counts a beat
that is overdue at that moment (``gateway/ws.py``). The supervisor's liveness watchdog thread
(``server_liveness``), which never runs on that loop, reads the stamp on its own cadence through ``AsgiLateWatch``:
a beat overdue by at least one whole interval while browser sockets are open and the server is not stopping is
one ``asgi_loop_late`` row per episode in ``logs/supervisor.jsonl``, with the loop thread's stack sampled once.
The loop writes and the watchdog reads without a lock (each field has one writer); nothing here touches the
disk on the loop, and nothing decides: it is the evidence a reader of the next "Reconnecting…" lacked.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class AsgiBeat:
    """The heartbeat's stamp of the ASGI loop, written only by the heartbeat coroutine on its loop.

    ``stamp``: monotonic time of the last beat, None while no heartbeat runs; ``late``: how much later than
    ``interval`` that beat came; ``ident``: the loop thread; ``interval``: the beat the heartbeat sleeps."""

    __slots__ = ("stamp", "late", "ident", "interval", "_owner")

    def __init__(self) -> None:
        self.stamp: Optional[float] = None
        self.late = 0.0
        self.ident = 0
        self.interval = 0.0
        self._owner: object = None

    def start(self, interval: float) -> object:
        """On the loop, as a heartbeat starts: this thread beats every ``interval`` seconds from now on."""
        token = object()
        self.ident, self.interval, self.late, self._owner = threading.get_ident(), float(interval), 0.0, token
        self.stamp = time.monotonic()
        return token

    def beat(self, slept_since: float) -> float:
        """On the loop, right after the sleep that began at ``slept_since``: stamp this beat; returns its lateness."""
        now = time.monotonic()
        self.late = max(0.0, now - slept_since - self.interval)
        self.stamp = now
        return self.late

    def stop(self, token: object) -> None:
        """The heartbeat that started with ``token`` ended; a later heartbeat's stamp stays."""
        if self._owner is token:
            self.stamp = None

    def overdue(self, now: float) -> float:
        """How far past due the next beat is at ``now``: 0 while it is on time or no heartbeat runs."""
        stamp = self.stamp
        return 0.0 if stamp is None else max(0.0, now - stamp - self.interval)


ASGI_BEAT = AsgiBeat()


class AsgiLateWatch:
    """The liveness watchdog's view of an ``AsgiBeat``.

    ``tick`` returns one ``asgi_loop_late`` row the first time it sees the beat overdue by at least one whole
    interval (a beat missed: the heartbeat's own interval, no other number) while ``has_clients`` says browser
    sockets are open and the server is not stopping, else None. An episode is one beat stamp: the next beat
    ends it, so a stamp is reported at most once. The row carries ``overdue_sec``; the loop thread's stack from
    the watchdog's code-only walker ``stack`` (bounded like a stall row; ``stack_truncated`` when cut); and
    ``resumed``, true when the loop beat again before its stack was taken, so that stack shows where the loop
    went on, not where it waited. No row while the beat is on time, no socket is open or nothing beats."""

    def __init__(self, stack: Callable[..., list], has_clients: Callable[[], bool], beat: AsgiBeat = ASGI_BEAT) -> None:
        self._stack, self._has_clients, self._beat = stack, has_clients, beat
        self._reported: Optional[float] = None

    def tick(self, now: float) -> Optional[dict]:
        beat = self._beat
        stamp, interval = beat.stamp, beat.interval
        if stamp is None or stamp == self._reported or now - stamp - interval < interval:
            return None
        from ouroboros.server_process import _exit_signalled, _restart_requested

        if _exit_signalled.is_set() or _restart_requested.is_set() or not self._has_clients():
            return None
        flags: dict = {}
        stack = self._stack(beat.ident, facts=flags)
        self._reported = stamp
        return {"type": "asgi_loop_late", "overdue_sec": round(now - stamp - interval, 1),
                "resumed": beat.stamp != stamp, **({"stack_truncated": True} if flags else {}),
                **({"stack": stack} if stack else {})}
