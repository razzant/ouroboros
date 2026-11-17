"""The supervisor consumer's load instruments, in memory only.

The supervisor has one consumer thread (the ``server.py`` loop) and one process-wide
queue lock Q (``supervisor.queue._queue_lock``). These windows let the one compact
``supervisor_load`` row per busy minute (``ouroboros/server_liveness.py``) say where the
consumer's time went: per event type, how many events its handlers served and their total
and longest service time; per bus lane, the worst producer-stamp lag the drain saw; for Q,
how long its outermost acquisitions waited and held it, from which code site the window's
longest hold came, and per code site the count, total and longest hold (``hold_by_site``: the
sites are the lexical acquisitions of Q, so the table is bounded by the code, and it is what
says whether a minute's hold time is one long barrier or many ordinary holds slowed by the
machine; ``machine_load`` adds the load averages and cores). Each window is a measurement only: nothing here touches disk,
a store or the event bus, decides anything or can raise into the code it measures, and its
one reader drains it (returns it and starts the next). Stdlib and the event taxonomy only,
so the queue module builds Q from it at import.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

from supervisor.event_taxonomy import disposition_for

_REPO_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _relative(filename: str) -> str:
    if filename.startswith(_REPO_PREFIX):
        filename = filename[len(_REPO_PREFIX):]
    return filename.replace(os.sep, "/")


def code_site(site: Optional[Tuple[str, int, str]]) -> Optional[str]:
    """``path:line in function`` (repository files relative), or None when no site was seen."""
    if not site:
        return None
    filename, lineno, name = site
    return f"{_relative(filename)}:{lineno} in {name}"


def code_place(place: Optional[Tuple[str, str]]) -> str:
    """``path in function``: a site without its line, the key of the per-site table."""
    if not place:
        return "unknown"
    filename, name = place
    return f"{_relative(filename)} in {name}"


class _LockWindow:
    """Outermost holds of one lock that ended in the window: their waits, holds, longest site, and
    per site (the function that took the lock, whatever the line) the count, total and longest hold."""

    __slots__ = ("count", "wait_total", "wait_max", "hold_total", "hold_max", "hold_max_site", "sites")

    def __init__(self) -> None:
        self.count = 0
        self.wait_total = self.wait_max = self.hold_total = self.hold_max = 0.0
        self.hold_max_site: Optional[Tuple[str, int, str]] = None
        self.sites: Dict[Optional[Tuple[str, str]], list] = {}  # (file, function) -> [count, hold_total, hold_max, wait_total]

    def note(self, waited: float, held: float, site: Optional[Tuple[str, int, str]]) -> None:
        self.count += 1
        self.wait_total += waited
        self.hold_total += held
        if waited > self.wait_max:
            self.wait_max = waited
        if held >= self.hold_max:  # a window whose holds all measure 0.0 still names a site
            self.hold_max, self.hold_max_site = held, site
        key = (site[0], site[2]) if site else None
        row = self.sites.get(key)
        if row is None:
            row = self.sites[key] = [0, 0.0, 0.0, 0.0]
        row[0] += 1
        row[1] += held
        row[3] += waited
        if held > row[2]:
            row[2] = held

    def facts(self) -> Dict[str, Any]:
        by_site = sorted(self.sites.items(), key=lambda item: -item[1][1])  # the heaviest total first
        return {
            "count": self.count,
            "wait_total_sec": round(self.wait_total, 3), "wait_max_sec": round(self.wait_max, 3),
            "hold_total_sec": round(self.hold_total, 3), "hold_max_sec": round(self.hold_max, 3),
            "hold_max_site": code_site(self.hold_max_site),
            "hold_by_site": {code_place(key): {
                "count": count, "hold_total_sec": round(total, 3), "hold_max_sec": round(longest, 3),
                "wait_total_sec": round(waited, 3)}
                for key, (count, total, longest, waited) in by_site},
        }


class MeasuredRLock:
    """Q: a ``threading.RLock`` that also times its OUTERMOST acquisitions.

    The lock semantics are the wrapped RLock's own (``with``, ``acquire(blocking, timeout)``,
    ``release`` raising the RLock's RuntimeError for a thread that does not hold it,
    ``_is_owned``), so every existing caller keeps working unchanged. An outermost
    acquisition keeps how long it waited before the grant and its caller's code site; its
    matching release records that wait and the hold in the current window. A re-entrant
    acquisition only counts depth; a refused non-blocking or timed-out attempt records
    nothing. The owner-only depth and stamps are guarded by Q itself, the window by one
    small lock taken only while Q is held or by ``drain``, never before Q: the reader never
    waits for Q, since an instrument that took Q would measure itself.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0
        self._waited = self._granted = 0.0
        self._site: Optional[Tuple[str, int, str]] = None
        self._stats = threading.Lock()
        self._window = _LockWindow()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._acquire(blocking, timeout)

    def __enter__(self) -> bool:
        return self._acquire(True, -1)

    def __exit__(self, *_exc: Any) -> None:
        self.release()

    def _acquire(self, blocking: bool, timeout: float) -> bool:
        if self._lock._is_owned():
            self._lock.acquire()  # re-entrant: granted at once, not measured again
            self._depth += 1
            return True
        asked = time.perf_counter()  # high resolution: monotonic ticks at ~15.6 ms on Windows
        if not self._lock.acquire(blocking, timeout):
            return False
        self._granted = time.perf_counter()
        self._waited, self._depth = self._granted - asked, 1
        try:
            frame = sys._getframe(2)  # 0 this method, 1 acquire/__enter__, 2 the caller
            self._site = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        except ValueError:
            self._site = None
        finally:
            frame = None  # never keep a caller's frame alive
        return True

    def release(self) -> None:
        try:
            if self._lock._is_owned():
                self._depth -= 1
                if not self._depth:
                    held = time.perf_counter() - self._granted
                    with self._stats:
                        self._window.note(self._waited, held, self._site)
                    self._site = None
        finally:
            self._lock.release()  # a thread that does not hold Q gets the RLock's own error

    def _is_owned(self) -> bool:
        return self._lock._is_owned()

    def drain(self) -> Dict[str, Any]:
        """The window's facts (holds that ended since the last drain), then a fresh window."""
        with self._stats:
            window, self._window = self._window, _LockWindow()
        return window.facts()

    def __repr__(self) -> str:
        return f"<MeasuredRLock {self._lock!r}>"


class ConsumerWindow:
    """The consumer's handler service time per event type and worst lag per lane.

    ``drain_worker_events`` writes it and the row reads it, both on the loop thread, so it
    takes no lock. The type key is bounded by the declared taxonomy: a kind without a
    disposition folds into ``undeclared``, so a producer cannot grow the row.
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self._handlers: Dict[str, list] = {}
        self._lag: Dict[str, float] = {}

    def note(self, kind: str, lane: str, seconds: float, lag: Optional[float]) -> None:
        key = kind if disposition_for(kind) is not None else "undeclared"
        stat = self._handlers.get(key)
        if stat is None:
            stat = self._handlers[key] = [0, 0.0, 0.0]
        stat[0] += 1
        stat[1] += seconds
        stat[2] = max(stat[2], seconds)
        if lag is not None and lag > self._lag.get(lane, float("-inf")):
            self._lag[lane] = lag

    def served(self) -> bool:
        """Whether the consumer served any event since the last drain."""
        return bool(self._handlers)

    def drain(self, now: float) -> Dict[str, Any]:
        """``window_sec``, ``handlers`` and ``lag`` (lane -> worst seconds) since the last drain."""
        facts = {
            "window_sec": round(now - self.started, 1),
            "handlers": {kind: {"count": count, "total_sec": round(total, 3), "max_sec": round(longest, 3)}
                         for kind, (count, total, longest) in sorted(self._handlers.items())},
            "lag": {lane: round(lag, 1) for lane, lag in self._lag.items()},
        }
        self.started, self._handlers, self._lag = now, {}, {}
        return facts


def machine_load() -> Dict[str, Any]:
    """The machine this minute, so a slow minute reads against what else ran: the 1-, 5- and 15-minute load
    averages (None where the OS keeps none, as on Windows: unknown, never 0) and the logical CPU count."""
    try:
        load = [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {"load_avg": load, "cpus": os.cpu_count()}


CONSUMER = ConsumerWindow()
