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
outside the store modules. ``drain()`` returns it as ``{kind: {"read": {...}, "write": {...}}}`` (only
buckets with a count) and resets it: the per-minute row in the journal whose tail every context reads.

The site table is a window of its own, because it weighs what the code's call paths weigh (36 to 42 KB
for a busy minute on the stand), not what a row a minute in that journal can: every site that held a
lock, most summed hold first, each with the same counts. It is bounded by the distinct sites the code
has, not by a cut: when one reader is hidden by a cut, its share is what nobody can read (40 to 48 of 43
to 51 sites were invisible behind a top three during the 01.11 saturation). A site's ``chains`` attribute
it to who called it: one ``{"chain", "count", "total_sec"}`` per distinct call chain, whose tag
(``_call_chain``) is the thread's class and the site's nearest first-party callers, so a site reached by
a duty, a request and a turn says how much each of them spent. ``drain_sites()`` returns it as
``{kind: {"read": [...], "write": [...]}}`` and resets it, on its publisher's own cadence
(``server_liveness._publish_store_sites``). Nothing here writes to disk, takes a store lock or raises
into the caller: a failure inside the hook is swallowed and the transaction's outcome stands.

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
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, Tuple

KINDS = {"current.sqlite": "current", "dialog.sqlite": "dialog", "usage.sqlite": "usage"}
# The frames the site skips: the store transport, not the owner that opened the transaction.
_STORE_FRAMES = frozenset({"store.py", "store_timing.py", "store_hot_facts.py", "store_task_cards.py", "task_store.py",
                           "dialogue_store.py", "usage_store.py", "store_capture.py", "contextlib.py"})
# A call chain's tag names this many first-party callers above the site: its shape, enough to tell the callers
# of one site apart in a row a minute (the thread's class says which duty or request reached them), not a policy.
CHAIN_CALLERS = 2
_OWN_PACKAGES = frozenset({"ouroboros", "supervisor", "server", "__main__"})  # whose frames can name a caller
_INSTANCE_TOKEN = re.compile(r"[-_ ]?[^-_ ]*\d[^-_ ]*")  # ``asyncio_12``, ``wake-turn-8f3a``: an instance, not a class
_LOCK = threading.Lock()
_WINDOW: Dict[str, Dict[str, Dict[str, Any]]] = {}
_SITES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}  # kind -> side -> site -> counts and chains
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
            site, chain = _site()
        except Exception:  # a site walk that fails still counts the hold
            site, chain = "", ""
        hold = "write" if write else "read"
        with _LOCK:
            bucket = _WINDOW.setdefault(kind, {}).setdefault(hold, {**_bucket(), "max_site": ""})
            at_site = (_SITES.setdefault(kind, {}).setdefault(hold, {}).setdefault(site, {**_bucket(), "chains": {}})
                       if site else None)
            for counts in (bucket, *((at_site,) if at_site is not None else ())):
                counts["count"] += 1
                counts["failed"] += int(failed)
                counts["total_sec"] += held
                counts["cpu_sec"] += cpu
                counts["max_sec"] = max(counts["max_sec"], held)
            if at_site is not None:
                by_chain = at_site["chains"].setdefault(chain, {"count": 0, "total_sec": 0.0})
                by_chain["count"] += 1
                by_chain["total_sec"] += held
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


def _site() -> Tuple[str, str]:
    """The site that opened this transaction, the first frame outside the store modules, and its call chain."""
    frame = sys._getframe(1)
    while frame is not None and os.path.basename(frame.f_code.co_filename) in _STORE_FRAMES:
        frame = frame.f_back
    if frame is None:
        return "", ""
    return f"{frame.f_globals.get('__name__', '?')}:{frame.f_code.co_name}:{frame.f_lineno}", _call_chain(frame.f_back)


def _call_chain(frame: Any) -> str:
    """Who reached a site: ``<thread class>: <caller> < <its caller>``, innermost first. The thread's class is its
    name without instance tokens (a duty's thread carries the duty's name, ``asyncio_12`` is the HTTP executor); a
    caller is a first-party ``module:function``, the module's last name kept (the stdlib and the dependencies
    between two first-party frames name nobody). Distinct chains are bounded by the code's call paths."""
    callers = []
    while frame is not None and len(callers) < CHAIN_CALLERS:
        module = str(frame.f_globals.get("__name__", ""))
        if module.partition(".")[0] in _OWN_PACKAGES:
            callers.append(f"{module.rpartition('.')[2]}:{frame.f_code.co_name}")
        frame = frame.f_back
    return f"{_INSTANCE_TOKEN.sub('', threading.current_thread().name)}: {' < '.join(callers)}"


def _rounded(counts: Dict[str, Any]) -> Dict[str, Any]:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in counts.items()}


def _published_site(site: str, counts: Dict[str, Any]) -> Dict[str, Any]:
    chains = sorted(counts["chains"].items(), key=lambda item: -item[1]["total_sec"])
    return {"site": site, **_rounded({key: value for key, value in counts.items() if key != "chains"}),
            "chains": [{"chain": chain, **_rounded(facts)} for chain, facts in chains]}


def drain() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """The window since the last drain (seconds rounded to microseconds), then an empty window."""
    with _LOCK:
        window = {kind: {side: _rounded(bucket) for side, bucket in sides.items()} for kind, sides in _WINDOW.items()}
        _WINDOW.clear()
    return window


def drain_sites() -> Dict[str, Dict[str, list]]:
    """Every site since the last drain of this table, most summed hold first, with its call chains; then an empty table."""
    with _LOCK:
        table = {kind: {side: [_published_site(site, counts) for site, counts in sorted(
            sites.items(), key=lambda item: -item[1]["total_sec"])] for side, sites in sides.items()}
            for kind, sides in _SITES.items()}
        _SITES.clear()
    return table
