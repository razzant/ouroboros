"""Wall and thread CPU time of each host section of one context build, for that build's own receipt.

A task's or a call's start is mostly the host sections its context reads, one after another. A builder
marked ``timed_section`` (or called through ``timed``) adds its durations to the spans of the build
measured on this thread of control (a context variable the agent opens around ``build_llm_messages``);
the ``context_memory_view`` receipt in ``logs/events.jsonl`` carries them as ``sections_ms`` and
``sections_cpu_ms``, with ``total``: the build's own time up to that receipt, so time no marked section
covers shows as the difference. Wall time far above CPU time is waiting (a lock, the disk, the
interpreter's other threads), so a slow start is read from the log instead of sampled. Spans are inclusive
and only for the marked builders; a dotted name (``memory_view.story``) is a span inside its parent, the time the
next reading must not sum in with it. Outside a measured build a marked builder costs one variable lookup and
records nothing.

``counted`` is the same receipt for work instead of time: the host adds what it did (a render, a parsed
record, a scanned character) to the window it runs in, and work done again on the same key also adds to
``<verb>_again``, so "done once" reads as a zero in the log and an equality in a test. ``section_spans``
carries the counts as ``work`` beside the two time maps. A window opens around any block that owes one
receipt (a context build, a round's preparation); windows nest and never mix.

``host_step`` marks where one named step of a round's window ends; ``llm_observability.round_window`` opens
that window and receipts the steps on the round's ``llm_round``. ``carry_spans`` names there the sections of a window
measured before it (a resumed turn's prologue: restore, route rebind, tools).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import gc
import threading
import time
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Set, Tuple

_SPANS: contextvars.ContextVar[Optional[Dict[str, List[float]]]] = contextvars.ContextVar("context_section_spans", default=None)
_STARTED: contextvars.ContextVar[Optional[Tuple[float, float]]] = contextvars.ContextVar("context_build_started", default=None)
_WORK: contextvars.ContextVar[Optional[Tuple[Dict[str, int], Set[Tuple[str, Hashable]]]]] = contextvars.ContextVar("context_build_work", default=None)
STEPS: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("round_host_steps", default=None)
TOTAL = "total"


def timed_section(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Add the builder's wall and thread CPU time under ``name`` to the build being measured, if any."""
    def decorate(builder: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(builder)
        def measured(*args: Any, **kwargs: Any) -> Any:
            spans = _SPANS.get()
            if spans is None:
                return builder(*args, **kwargs)
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return builder(*args, **kwargs)
            finally:
                span = spans.setdefault(name, [0.0, 0.0])
                span[0] += time.perf_counter() - wall
                span[1] += time.thread_time() - cpu
        return measured
    return decorate


def timed(name: str, builder: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``builder(*args, **kwargs)`` measured under ``name``, for a builder whose module does not import this one."""
    return timed_section(name)(builder)(*args, **kwargs)


def counted(verb: str, n: int = 1, key: Optional[Hashable] = None) -> None:
    """Add ``n`` to ``verb`` of the window being measured, if any; a ``key`` this verb already saw in that
    window adds ``n`` to ``<verb>_again`` as well."""
    window = _WORK.get()
    if window is None:
        return
    work, seen = window
    work[verb] = work.get(verb, 0) + n
    if key is None:
        return
    if (verb, key) in seen:
        work[f"{verb}_again"] = work.get(f"{verb}_again", 0) + n
    seen.add((verb, key))


def host_step(name: str) -> None:
    """The round's step ``name`` ends here: its wall and thread CPU time since the previous mark, or since the window
    opened, add to its span (nothing outside a window). A step that ends on another thread than it began (the model
    sender's, which runs in a copy of this context) has no one thread's CPU time: unknown (None)."""
    state = STEPS.get()
    if state is None:
        return
    now = (time.perf_counter(), time.thread_time(), threading.get_ident())
    wall, cpu, thread = state["mark"]
    span = state["steps"].setdefault(name, [0.0, 0.0])
    span[0] += now[0] - wall
    span[1] = None if span[1] is None or now[2] != thread else span[1] + now[1] - cpu
    state["mark"] = now


@contextlib.contextmanager
def measured_sections() -> Iterator[None]:
    """Measure the marked sections and count the work of this block: one window, one receipt."""
    token, started, work = _SPANS.set({}), _STARTED.set((time.perf_counter(), time.thread_time())), _WORK.set(({}, set()))
    try:
        yield
    finally:
        _WORK.reset(work)
        _STARTED.reset(started)
        _SPANS.reset(token)


_REST = threading.Lock()
_rest = [0, False]  # context builds resting the collector now; whether it ran when the first of them began


@contextlib.contextmanager
def collector_rests() -> Iterator[None]:
    """The cyclic collector waits while a context build runs and resumes in ``finally``: the build allocates millions of
    short-lived objects and creates no cycles, so each collection only walks the process's long-lived heap again.
    Builds overlap in one process (the server's inline turns), so the rest is counted: the last one out restores it.
    Measured on a copy of the owner's data: peak RSS +32…+119 MB during a build, CPU −0.1…−2.5 s."""
    with _REST:
        _rest[0] += 1
        if _rest[0] == 1:
            _rest[1] = gc.isenabled()
            gc.disable()
    try:
        yield
    finally:
        with _REST:
            _rest[0] -= 1
            if not _rest[0] and _rest[1]:
                gc.enable()


def mark() -> Tuple[float, float, int]:
    """Now, for ``span_since``: wall clock, this thread's CPU clock and this thread."""
    return time.perf_counter(), time.thread_time(), threading.get_ident()


def span_since(name: str, started: Tuple[float, float, int]) -> None:
    """Add the time since ``started`` (``mark``) under ``name`` to the window being measured, if any: for work another
    owner does between two calls of this one (a ledger's reservation). Ending on another thread, it adds no CPU time."""
    spans = _SPANS.get()
    if spans is None:
        return
    span = spans.setdefault(name, [0.0, 0.0])
    span[0] += time.perf_counter() - started[0]
    span[1] += time.thread_time() - started[1] if threading.get_ident() == started[2] else 0.0


@contextlib.contextmanager
def apart() -> Iterator[Dict[str, Any]]:
    """A window of its own for work run beside the measured one, on another thread: what it counts stays in the
    yielded holder for ``adopt`` and is never written into the window it runs beside while that one is read."""
    holder: Dict[str, Any] = {}
    with measured_sections():
        try:
            yield holder
        finally:
            holder["window"] = _WORK.get()


@contextlib.contextmanager
def another_pass() -> Iterator[None]:
    """Work the measured window repeats by design (the fit's second route question and walk) as a pass of its own: its
    counts add to the window, a key it repeats itself counts ``_again``, and its keys join the window's afterwards, so
    only work done twice within one pass, or after both, reads as done again."""
    window = _WORK.get()
    if window is None:
        yield
        return
    keys: Set[Tuple[str, Hashable]] = set()
    token = _WORK.set((window[0], keys))
    try:
        yield
    finally:
        _WORK.reset(token)
        window[1].update(keys)


def adopt(holder: Optional[Dict[str, Any]]) -> None:
    """Add work counted ``apart`` (its counts and seen keys) to the window being measured, on that window's thread."""
    window, target = (holder or {}).get("window"), _WORK.get()
    if window is None or target is None:
        return
    target[1].update(window[1])
    for verb, n in window[0].items():
        target[0][verb] = target[0].get(verb, 0) + n


def carry_spans(prefix: str, receipt: Optional[Dict[str, Dict[str, int]]]) -> None:
    """Add the sections of a window measured before this one (its ``section_spans``) to the window being measured, each
    under ``prefix`` (its ``total`` as ``prefix`` itself): work done before a receipt's window opened, named once."""
    spans = _SPANS.get()
    if spans is None or not receipt:
        return
    cpu = receipt.get("sections_cpu_ms") or {}
    for name, wall in (receipt.get("sections_ms") or {}).items():
        span = spans.setdefault(prefix if name == TOTAL else f"{prefix}.{name}", [0.0, 0.0])
        span[0] += wall / 1000
        span[1] += cpu.get(name, 0) / 1000


def section_spans() -> Dict[str, Dict[str, int]]:
    """``sections_ms`` and ``sections_cpu_ms`` of the window being measured (with its ``total`` so far) and
    its counted ``work``, or nothing outside one."""
    spans, started, window = _SPANS.get(), _STARTED.get(), _WORK.get()
    if spans is None or started is None or window is None:
        return {}
    now = {TOTAL: [time.perf_counter() - started[0], time.thread_time() - started[1]]}
    return {"sections_ms": {name: round(span[0] * 1000) for name, span in {**spans, **now}.items()},
            "sections_cpu_ms": {name: round(span[1] * 1000) for name, span in {**spans, **now}.items()},
            "work": dict(window[0])}
