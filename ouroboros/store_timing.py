"""Lock timing of store transactions, in memory, per process.

``store.transaction`` calls ``record`` once per transaction, three phases timed on one clock: acquire
(from just before ``BEGIN`` until ``started()``: ``BEGIN IMMEDIATE`` returned, or a read's first SELECT
took its SHARED lock), body, and commit (its end on that connection: COMMIT, an error's ROLLBACK, a
``finish``). ``store.commit`` stamps its COMMIT on the connection (``ended``): that COMMIT is then the
commit phase and the hold ends where it returned (a failed one's where its ROLLBACK did), also when the
body ran it itself (the dispatch claim's zero-wait COMMIT; the body after it holds no lock, the ``finish``
has nothing left to end). Each is elapsed time on the calling thread (SQLite's busy-handler sleeps with
scheduling and CPU, not a lock wait alone); the hold is body and commit, a COMMIT's wait on draining
readers included; ``failed`` says it ended in an error (raised inside it, or a COMMIT that did not
complete). A lock refused busy is no hold but a refusal, its acquire counted; ``mark_busy`` puts on the
error the phase that met it, the seconds elapsed in that phase and the connection's ``busy_timeout``
(``None`` where unknown), which ``busy_note`` words for the typed error; ``ended`` marks a COMMIT's own,
before any caller types it (the money store does, inside the transaction). Who held the lock is not known
here: another thread's or process's holds are only its own rows. The window keeps, per store kind
(``current``, ``dialog``, ``usage``
or ``other``, from the connection's database file name; the dialogue and usage stores share
``store.transaction``) and per ``read``/``write``, totals: ``count`` (holds), ``failed``, ``total_sec``,
``cpu_sec`` (the holding thread's own CPU: wall minus it is time spent holding the lock while waiting for
the GIL, a core or a lock), ``commit_sec`` (the commit phases' part of ``total_sec``), ``acquire_sec``
(every acquire, refusals' included), ``refused``; maxima, each its own: ``max_sec`` (holds),
``max_acquire_sec``; and of the window's longest hold alone, its commit phase (``max_commit_sec``) and
code site (``max_site``, ``module:function:line``, the first frame outside the store modules).
``drain()`` returns it as ``{kind: {"read": {...}, "write": {...}}}`` (only buckets with a hold or a
refusal) and resets it: the per-minute row in the journal whose tail every context reads.

The site table is a window of its own, because it weighs what the code's call paths weigh (36 to 42 KB
for a busy minute on the stand), not what a row a minute in that journal can: every site that held or
was refused a lock, most summed hold first, each with the same counts but the longest hold's. It is
bounded by the distinct sites the code has, not by a cut: when one reader is hidden by a cut, its share
is what nobody can read (40 to 48 of 43 to 51 sites were invisible behind a top three during the 01.11
saturation). A site's ``chains`` attribute its holds to who called it: one ``{"chain", "count",
"total_sec"}`` per distinct call chain, whose tag (``_call_chain``) is the thread's class and the site's
nearest first-party callers, so a site reached by a duty, a request and a turn says how much each of them
spent. ``drain_sites()`` returns it as
``{kind: {"read": [...], "write": [...]}}`` and resets it, on its publisher's own cadence
(``server_liveness._publish_store_sites``). Nothing here writes to disk, takes a store lock or raises
into the caller: a failure inside the hook is swallowed and the transaction's outcome stands.

``metered()`` scopes a second, private count to one unit of work on the calling context (an owner
turn on its thread; a ``ContextVar``, the ``store._WAIT_BUDGET`` idiom): the same transactions as
``{kind}.{read|write}`` with their summed and longest holds (``….sec``, ``….max_sec``) and summed
acquire phases, refusals' included (``….acquire_sec``), plus the events ``count`` names (the chronicle
index's opens, ``chronicle.locked``; a lock-free read counts ``chronicle.read``). ``snapshot`` copies
the meter by value for a receipt read from another thread.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

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


def started(conn: Any) -> tuple:
    """When a transaction's lock was taken on ``conn``, as ``record`` measures it: ``(clock(), cpu_clock())``. An
    end ``ended`` stamped on ``conn`` before belongs to an earlier transaction and is forgotten."""
    with contextlib.suppress(Exception):  # a connection without attributes holds no stamp to forget
        conn._store_end = None
    return clock(), cpu_clock()


def _bucket() -> Dict[str, Any]:
    return {"count": 0, "failed": 0, "total_sec": 0.0, "cpu_sec": 0.0, "max_sec": 0.0, "commit_sec": 0.0,
            "acquire_sec": 0.0, "max_acquire_sec": 0.0, "refused": 0}


def kind_of(path: Any) -> str:
    """The store kind a database file name stands for (``other`` for anything else)."""
    return KINDS.get(os.path.basename(os.fspath(path)), "other")


def record(conn: Any, write: bool, since: Optional[tuple], *, asked: float, ending: Optional[float],
           failed: bool = False) -> None:
    """One transaction's lock, asked for at ``asked`` (``clock()`` before ``BEGIN``), held from ``since``
    (``started()``) until now, its end (COMMIT or ROLLBACK) begun at ``ending`` (``clock()``); a COMMIT ``ended``
    stamped on ``conn`` since is its end instead, the hold ending where that returned. ``failed`` when it ended in
    an error. ``since=None``: the lock was refused busy at ``ending`` (now if ``None``), a refusal and no hold.
    Never raises. A hold as long as the window's longest names its site and end too, so a window of 0.0 holds
    still names one."""
    try:
        now, cpu_now = clock(), cpu_clock()
        stamp = getattr(conn, "_store_end", None) if since is not None else None
        if stamp:
            ending, now, cpu_now = stamp
        acquire = max(0.0, (since[0] if since is not None else now if ending is None else ending) - asked)
        held = 0.0 if since is None else now - since[0]
        cpu = 0.0 if since is None else max(0.0, cpu_now - since[1])
        end = 0.0 if since is None or ending is None else max(0.0, now - ending)
        kind = getattr(conn, "_store_kind", None) or "other"
        try:
            site, chain = _site()
        except Exception:  # a site walk that fails still counts the hold
            site, chain = "", ""
        hold = "write" if write else "read"
        with _LOCK:
            bucket = _WINDOW.setdefault(kind, {}).setdefault(hold, {**_bucket(), "max_commit_sec": 0.0, "max_site": ""})
            at_site = (_SITES.setdefault(kind, {}).setdefault(hold, {}).setdefault(site, {**_bucket(), "chains": {}})
                       if site else None)
            for counts in (bucket, *((at_site,) if at_site is not None else ())):
                counts["acquire_sec"] += acquire
                counts["max_acquire_sec"] = max(counts["max_acquire_sec"], acquire)
                if since is None:
                    counts["refused"] += 1
                    continue
                counts["count"] += 1
                counts["failed"] += int(failed)
                counts["total_sec"] += held
                counts["cpu_sec"] += cpu
                counts["commit_sec"] += end
                counts["max_sec"] = max(counts["max_sec"], held)
            if at_site is not None and since is not None:
                by_chain = at_site["chains"].setdefault(chain, {"count": 0, "total_sec": 0.0})
                by_chain["count"] += 1
                by_chain["total_sec"] += held
            if since is not None and held >= bucket["max_sec"]:
                bucket["max_site"], bucket["max_commit_sec"] = site, end
            meter = _METER.get()
            if meter is not None:
                side = f"{kind}.{hold}"
                meter[side + ".acquire_sec"] = meter.get(side + ".acquire_sec", 0.0) + acquire
                if since is not None:
                    meter[side] = meter.get(side, 0) + 1
                    meter[side + ".sec"] = meter.get(side + ".sec", 0.0) + held
                    meter[side + ".max_sec"] = max(meter.get(side + ".max_sec", 0.0), held)
    except Exception:  # a diagnostic never changes the transaction's outcome
        pass


def mark_busy(exc: BaseException, conn: Any, *, asked: float, held: Optional[tuple], ending: Optional[float],
              failed_at: float) -> None:
    """Put on a busy error met at ``failed_at`` the transaction phase that met it (``acquire``: no lock ``held``
    yet, ``BEGIN`` or a read's first SELECT; ``commit``: its end on this connection had begun at ``ending``; else
    ``body``), the seconds elapsed in that phase (busy-handler sleeps with scheduling and CPU) and the connection's
    ``busy_timeout`` (seconds), read only now; ``None`` is unknown: a busy statement inside the body may have run on
    another connection. An inner transaction's marks stand, as do a COMMIT's own (``ended``). Never raises."""
    try:
        phase = "acquire" if held is None else "body" if ending is None else "commit"
        _mark(exc, conn, phase, None if phase == "body" else failed_at - (asked if phase == "acquire" else ending))
    except Exception:
        pass


def ended(conn: Any, began: float, busy: Optional[BaseException] = None) -> None:
    """``store.commit``'s COMMIT on ``conn``, begun at ``began`` (``clock()``), has returned: committed, or failed
    and rolled back. Stamped on ``conn`` for its transaction's ``record``; a ``busy`` error is marked ``commit``
    with the seconds since ``began`` and the ``busy_timeout``, before any caller types it. Never raises."""
    try:
        now = clock()
        with contextlib.suppress(Exception):  # a connection without attributes: its transaction's own end stands
            conn._store_end = (began, now, cpu_clock())
        if busy is not None:
            _mark(busy, conn, "commit", now - began)
    except Exception:
        pass


def failed(exc: BaseException, conn: Any, write: bool, held: Optional[tuple], *, asked: float,
           ending: Optional[float], failed_at: float, contended: bool) -> None:
    """A transaction ended in ``exc`` at ``failed_at``: a failed hold when its lock was ``held``; a refusal, its
    acquire counted, when the lock was refused busy (``contended``); nothing for another error before the lock.
    A busy error carries its phase (``mark_busy``). Never raises."""
    if held:
        record(conn, write, held, asked=asked, ending=failed_at if ending is None else ending, failed=True)
    elif contended:
        record(conn, write, None, asked=asked, ending=failed_at)
    if contended:
        mark_busy(exc, conn, asked=asked, held=held, ending=ending, failed_at=failed_at)


def _mark(exc: BaseException, conn: Any, phase: str, elapsed: Optional[float]) -> None:
    """Put ``phase``, ``elapsed`` and ``conn``'s ``busy_timeout`` (read now; not for the body) on ``exc``, unless
    marks are on it already."""
    if getattr(exc, "store_phase", None) is not None:
        return
    timeout = None
    if phase != "body":
        try:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0] / 1000.0
        except Exception:  # an unreadable budget stays unknown
            timeout = None
    exc.store_phase, exc.store_elapsed_sec, exc.store_busy_timeout_sec = phase, elapsed, timeout


def busy_note(exc: BaseException) -> str:
    """``" [acquire: elapsed 0.000 s, busy_timeout 0 s]"`` for a busy error ``mark_busy`` marked, else ``""``."""
    try:
        phase = getattr(exc, "store_phase", None)
        if not phase:
            return ""
        elapsed, timeout = getattr(exc, "store_elapsed_sec", None), getattr(exc, "store_busy_timeout_sec", None)
        return (f" [{phase}: elapsed " + ("unknown" if elapsed is None else f"{elapsed:.3f} s")
                + ", busy_timeout " + ("unknown" if timeout is None else f"{timeout:g} s") + "]")
    except Exception:
        return ""


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
