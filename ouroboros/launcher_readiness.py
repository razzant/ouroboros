"""The launcher's ``/api/health`` probe of a server it waits for.

The server answers 200 as soon as its socket serves, during boot too; the probe logs the ``boot``
phase the body reports and leaves the rest to the UI. It has no clock for the launcher's own child,
which it waits for as long as that child lives.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

log = logging.getLogger("launcher")


def wait_for_server(port: int, timeout: Optional[float] = None, abort_event=None, proc=None) -> bool:
    """Wait for the agent HTTP server to answer ``/api/health`` (200 as soon as its socket serves;
    the body's ``boot`` phase is logged, the UI shows the rest).

    The wait ends with the first answer, ``proc``'s exit or ``abort_event`` (headless mode: a
    shutdown signal only sets the event, so the wait itself owns the early exit). ``timeout`` is a
    bound for probing a server this launcher does not own (the already-running notice); the
    launcher's own child is waited for as long as it lives. The per-probe HTTP timeout stays: it is
    a transport bound, not a verdict."""
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = None if timeout is None else time.time() + timeout
    while deadline is None or time.time() < deadline:
        if abort_event is not None and abort_event.is_set():
            return False
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    _log_boot_phase(response)
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _log_boot_phase(response) -> None:
    """One line about the phases the serving generation reports (older cores report none)."""
    try:
        boot = json.loads(response.read().decode("utf-8")).get("boot")
    except Exception:
        return
    if isinstance(boot, dict):
        log.info("Agent server is serving: boot phase %s, dialogue_ready=%s, tasks_ready=%s",
                 boot.get("phase"), boot.get("dialogue_ready"), boot.get("tasks_ready"))
