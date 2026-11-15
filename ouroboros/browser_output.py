"""Playwright driver diagnostics join the process buffer; its stdout protocol stays exact.

Playwright has no public stderr argument. Its pinned transport asks a small getter in the same
loop step that creates the driver process. Adapt that getter only inside a ``driver_diagnostics``
scope: for the asyncio task that starts a browser generation's driver and the tasks it creates,
which copy its context (the connection's runner), never another coroutine of that loop; or for the
thread running ``sync_playwright``, whose dispatcher greenlet runs on that thread in a context of
its own. Outside every scope, and once its scope has closed, the original getter answers: a spawn
the scope did not reach inherits stderr as before, never a closed descriptor. An already-started
child owns its pipe handle until exit. No process descriptor or asyncio spawn function is replaced.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import threading
from types import SimpleNamespace

from ouroboros.process_logging import relayed_child_output

_task_output: contextvars.ContextVar = contextvars.ContextVar("ouroboros_driver_output", default=None)
_thread_output = threading.local()


def _adapt_getter() -> None:
    """Concurrent installs are equivalent wrappers of the same original getter; no lock survives into a fork."""
    from playwright._impl import _transport

    original = _transport._get_stderr_fileno
    if getattr(original, "_ouroboros_diagnostics", False):
        return

    def stderr():
        for held in (_task_output.get(), getattr(_thread_output, "held", None)):
            if held is not None and held.open and held.pid == os.getpid():
                return held.fd
        return original()

    stderr._ouroboros_diagnostics = True
    _transport._get_stderr_fileno = stderr


@contextlib.contextmanager
def driver_diagnostics(*, task: bool = False):
    """Scope one driver start: ``task`` for the asyncio task that starts it, else the calling thread (sync API)."""
    with relayed_child_output() as output:
        if output is None:
            yield
            return
        _adapt_getter()
        held = SimpleNamespace(fd=output, pid=os.getpid(), open=True)
        if task:
            token = _task_output.set(held)
        else:
            previous, _thread_output.held = getattr(_thread_output, "held", None), held
        try:
            yield
        finally:
            held.open = False  # a task that copied this context keeps the value, never the descriptor
            if task:
                _task_output.reset(token)
            else:
                _thread_output.held = previous
