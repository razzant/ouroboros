"""Wall and thread CPU time of each host section of one context build, for that build's own receipt.

A task's or a call's start is mostly the host sections its context reads, one after another. A builder
marked ``timed_section`` (or called through ``timed``) adds its durations to the spans of the build
measured on this thread of control (a context variable the agent opens around ``build_llm_messages``);
the ``context_memory_view`` receipt in ``logs/events.jsonl`` carries them as ``sections_ms`` and
``sections_cpu_ms``, with ``total``: the build's own time up to that receipt, so time no marked section
covers shows as the difference. Wall time far above CPU time is waiting (a lock, the disk, the
interpreter's other threads), so a slow start is read from the log instead of sampled. Spans are inclusive
and only for the marked builders. Outside a measured build a marked builder costs one variable lookup and
records nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_SPANS: contextvars.ContextVar[Optional[Dict[str, List[float]]]] = contextvars.ContextVar("context_section_spans", default=None)
_STARTED: contextvars.ContextVar[Optional[Tuple[float, float]]] = contextvars.ContextVar("context_build_started", default=None)
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


@contextlib.contextmanager
def measured_sections() -> Iterator[None]:
    """Measure the sections of the context build inside this block."""
    token, started = _SPANS.set({}), _STARTED.set((time.perf_counter(), time.thread_time()))
    try:
        yield
    finally:
        _STARTED.reset(started)
        _SPANS.reset(token)


def section_spans() -> Dict[str, Dict[str, int]]:
    """``sections_ms`` and ``sections_cpu_ms`` of the build being measured (with its ``total`` so far),
    or nothing outside one."""
    spans, started = _SPANS.get(), _STARTED.get()
    if spans is None or started is None:
        return {}
    now = {TOTAL: [time.perf_counter() - started[0], time.thread_time() - started[1]]}
    return {"sections_ms": {name: round(span[0] * 1000) for name, span in {**spans, **now}.items()},
            "sections_cpu_ms": {name: round(span[1] * 1000) for name, span in {**spans, **now}.items()}}
