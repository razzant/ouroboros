"""Lock-hold timing of store transactions, in memory, per process.

``store.transaction`` calls ``record`` once per transaction with when its lock was taken
(``started()``: ``BEGIN IMMEDIATE`` returned, or a read's first SELECT took its SHARED lock) and
measures until COMMIT or ROLLBACK returned, a COMMIT's wait on draining readers included; ``failed``
says it ended in an error (raised inside it, or a COMMIT that did not complete). The window keeps,
per store kind (``current``, ``dialog``, ``usage`` or ``other``, from the connection's database file
name; the dialogue and usage stores share ``store.transaction``) and per ``read``/``write``:
``count``, ``failed``, ``total_sec``, ``cpu_sec`` (the holding thread's own CPU: wall minus it is
time spent holding the lock while waiting for the GIL, a core or a lock), ``max_sec`` and
``max_site``, the code site (``module:function:line``) of the window's longest hold, the first frame
outside the store modules, and ``sites``: the ``TOP_SITES`` sites with the most summed hold, each
with the same counts, with ``sites_omitted`` saying how many sites that cut left out of the row.
``drain()`` returns the window as ``{kind: {"read": {...}, "write": {...}}}``
(only buckets with a count) and resets it: the per-minute diagnostics row publishes it. Nothing here
writes to disk, takes a store lock or raises into the caller: a failure inside the hook is swallowed
and the transaction's outcome stands.

``metered()`` scopes a second, private count to one unit of work on the calling context (an owner
turn on its thread; a ``ContextVar``, the ``store._WAIT_BUDGET`` idiom): the same transactions as
``{kind}.{read|write}`` with their summed and longest holds (``….sec``, ``….max_sec``), plus the
events ``count`` names (the chronicle index's opens, ``chronicle.locked``; a lock-free read counts
``chronicle.read``). ``snapshot`` copies the meter by value for a receipt read from another thread.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
import time
from typing import Any, Dict, Iterator

KINDS = {"current.sqlite": "current", "dialog.sqlite": "dialog", "usage.sqlite": "usage"}
# The frames the site skips: the store transport, not the owner that opened the transaction.
_STORE_FRAMES = frozenset({"store.py", "store_timing.py", "store_hot_facts.py", "store_task_cards.py", "task_store.py",
                           "dialogue_store.py", "usage_store.py", "store_capture.py", "contextlib.py"})
TOP_SITES = 3  # sites per bucket in the published row: the row's size bound (one row a minute), not a policy
_LOCK = threading.Lock()
_WINDOW: Dict[str, Dict[str, Dict[str, Any]]] = {}
_METER: contextvars.ContextVar = contextvars.ContextVar("store_timing_meter", default=None)
# The hold clock: high resolution on every OS (``time.monotonic`` ticks at ~15.6 ms on Windows, where a short hold
# would measure 0.0); the holding thread's own CPU beside it.
clock = time.perf_counter
cpu_clock = time.thread_time


def started() -> tuple:
    """When a transaction's lock was taken, as ``record`` measures it: ``(clock(), cpu_clock())``."""
    return clock(), cpu_clock()


def _bucket() -> Dict[str, Any]:
    return {"count": 0, "failed": 0, "total_sec": 0.0, "cpu_sec": 0.0, "max_sec": 0.0}


def kind_of(path: Any) -> str:
    """The store kind a database file name stands for (``other`` for anything else)."""
    return KINDS.get(os.path.basename(os.fspath(path)), "other")


def record(conn: Any, write: bool, since: tuple, *, failed: bool = False) -> None:
    """One transaction held its lock from ``since`` (``started()``) until now, ``failed`` when it ended in
    an error; never raises. A hold as long as the window's longest names its site too, so a window whose holds
    all measure 0.0 still names one."""
    try:
        held, cpu = clock() - since[0], max(0.0, cpu_clock() - since[1])
        kind = getattr(conn, "_store_kind", None) or "other"
        try:
            site = _site()
        except Exception:  # a site walk that fails still counts the hold
            site = ""
        with _LOCK:
            bucket = _WINDOW.setdefault(kind, {}).setdefault("write" if write else "read", {**_bucket(), "max_site": "",
                                                                                              "sites": {}})
            for counts in (bucket, *((bucket["sites"].setdefault(site, _bucket()),) if site else ())):
                counts["count"] += 1
                counts["failed"] += int(failed)
                counts["total_sec"] += held
                counts["cpu_sec"] += cpu
                counts["max_sec"] = max(counts["max_sec"], held)
            if held >= bucket["max_sec"]:
                bucket["max_site"] = site
            meter = _METER.get()
            if meter is not None:
                side = f"{kind}.{'write' if write else 'read'}"
                meter[side] = meter.get(side, 0) + 1
                meter[side + ".sec"] = meter.get(side + ".sec", 0.0) + held
                meter[side + ".max_sec"] = max(meter.get(side + ".max_sec", 0.0), held)
    except Exception:  # a diagnostic never changes the transaction's outcome
        pass


def count(event: str) -> None:
    """One ``event`` inside the current ``metered()`` scope, if any; never raises."""
    try:
        meter = _METER.get()
        if meter is not None:
            with _LOCK:
                meter[event] = meter.get(event, 0) + 1
    except Exception:
        pass


@contextlib.contextmanager
def metered() -> Iterator[Dict[str, Any]]:
    """This context's own counts (module docstring); read them through ``snapshot`` from another thread."""
    meter: Dict[str, Any] = {}
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)


def snapshot(meter: Any) -> Dict[str, Any]:
    """A by-value copy of a meter (seconds rounded to microseconds); ``{}`` for no meter."""
    if not isinstance(meter, dict):
        return {}
    with _LOCK:
        return {key: round(value, 6) if isinstance(value, float) else value for key, value in meter.items()}


def _site() -> str:
    frame = sys._getframe(1)
    while frame is not None and os.path.basename(frame.f_code.co_filename) in _STORE_FRAMES:
        frame = frame.f_back
    return "" if frame is None else f"{frame.f_globals.get('__name__', '?')}:{frame.f_code.co_name}:{frame.f_lineno}"


def _rounded(counts: Dict[str, Any]) -> Dict[str, Any]:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in counts.items()}


def drain() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The window since the last drain (seconds rounded to microseconds; per bucket the ``TOP_SITES``
    sites by summed hold and the count of sites the cut left out), then an empty window."""
    with _LOCK:
        window = {kind: {side: {**_rounded({key: value for key, value in bucket.items() if key != "sites"}),
                                "sites": [{"site": site, **_rounded(counts)} for site, counts in sorted(
                                    bucket["sites"].items(), key=lambda item: -item[1]["total_sec"])[:TOP_SITES]],
                                "sites_omitted": max(0, len(bucket["sites"]) - TOP_SITES)}
                         for side, bucket in sides.items()} for kind, sides in _WINDOW.items()}
        _WINDOW.clear()
    return window
