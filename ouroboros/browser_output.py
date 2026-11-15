"""Playwright driver diagnostics join the process buffer; its stdout protocol stays exact.

Playwright has no public stderr argument. Its pinned transport asks a small
getter immediately before subprocess creation. Adapt that getter only for the
thread starting our driver, including the sync API's dispatcher greenlet. Other
threads keep the original getter; an already-started child owns its pipe handle
until exit. No process descriptor or asyncio spawn function is replaced.
"""
from __future__ import annotations

import contextlib
import os
import threading

from ouroboros.process_logging import relayed_child_output

_starting = threading.local()


@contextlib.contextmanager
def driver_diagnostics():
    """Scope only Playwright.start/enter on its owner thread, sync or async."""
    with relayed_child_output() as output:
        if output is None:
            yield
            return
        from playwright._impl import _transport

        # Concurrent installs are equivalent wrappers of the same original getter.
        # No lock survives into a forked child.
        original = _transport._get_stderr_fileno
        if not getattr(original, "_ouroboros_diagnostics", False):
            def stderr():
                held = getattr(_starting, "output", None)
                return held[1] if held and held[0] == os.getpid() else original()
            stderr._ouroboros_diagnostics = True
            _transport._get_stderr_fileno = stderr
        previous = getattr(_starting, "output", None)
        _starting.output = (os.getpid(), output)
        try:
            yield
        finally:
            _starting.output = previous
