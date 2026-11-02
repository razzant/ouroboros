"""Before/after observations on the one retained managed-update attempt.

Facts, never a verdict or chat notification. Samples run at update admission,
confirmed boot adoption, duty overruns and supervisor dispatch-event handling.
The turn's observer records memory only; no poller or history replay. Each side
holds one sample and the latest observed direct turn. The next update replaces
the record. Producer hooks never infer dispatch from an active heartbeat.
"""
from __future__ import annotations

import copy
import logging
import os
import pathlib
import sys
import threading
import time

from ouroboros.utils import read_json_dict, utc_now_iso

log = logging.getLogger(__name__)
_lock = threading.Lock()
_duties: dict[str, int] | None = None
_direct: dict | None = None
_dispatch_pending = False
_cpu_sample: tuple[float, float] | None = None


def _rss() -> tuple[int | None, str]:
    """In-process observation; a portable peak fallback is labelled as such.

    No subprocess or census on a direct/duty producer. psutil is shipped on
    Windows; POSIX's stdlib fallback explicitly reports the lifetime high-water
    mark, never presents it as current RSS.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss, "current"
    except ImportError:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak) * (1 if sys.platform == "darwin" else 1024), "peak"
        except (ImportError, OSError):
            return None, "unobserved"
    except Exception:
        return None, "unobserved"


def sample() -> dict:
    from ouroboros.gateway.update_progress import _generation
    from ouroboros.server_process import server_source_baseline

    global _cpu_sample
    now, cpu = time.monotonic(), time.process_time()
    with _lock:
        previous, _cpu_sample = _cpu_sample, (now, cpu)
        interval = now - previous[0] if previous else None
        cpu_percent = 100 * (cpu - previous[1]) / interval if interval and interval > 0 else None
        duties, direct = dict(_duties) if _duties is not None else None, copy.deepcopy(_direct)
    rss, rss_measure = _rss()
    return {"generation": _generation(), "source_sha": server_source_baseline(),
            "observed_at": utc_now_iso(), "pid": os.getpid(), "scope": "server_process",
            "rss_bytes": rss, "rss_measure": rss_measure, "cpu_seconds": cpu, "cpu_percent": cpu_percent,
            "cpu_interval_seconds": interval, "duty_overruns": duties,
            "direct_message_first_dispatch": direct}


def _publish_after() -> None:
    """Replace only this serving generation's sample of the retained applied attempt."""
    from ouroboros.gateway import update_progress

    try:
        # This is an event-side read, not a display mutation or an idle duty.
        status, record = update_progress._read_saved()
        if status != "valid" or record.get("outcome") != "applied":
            return
        after = (record.get("generation_facts") or {}).get("after") or {}
        if after.get("generation") != update_progress._generation():
            return
        observed = sample()

        def mutate(stored):
            if (stored.get("operation_id") != record.get("operation_id") or stored.get("outcome") != "applied"
                    or ((stored.get("generation_facts") or {}).get("after") or {}).get("generation") != observed["generation"]):
                return None
            return {**stored, "generation_facts": {**stored["generation_facts"], "after": observed}}

        update_progress._store(mutate)
    except Exception:
        log.debug("Could not retain after-update facts", exc_info=True)


def observe_duty_overrun(name: str) -> None:
    """The duty owner calls once when its existing overrun observation is emitted."""
    global _duties
    with _lock:
        if _duties is None:
            _duties = {}
        _duties[name] = _duties.get(name, 0) + 1
    _publish_after()


def observe_direct_dispatch(*, task_id: str, accepted_at: float, dispatched_at: float,
                            attempt_id: str = "") -> None:
    """The direct-turn owner calls at its first actual model dispatch, once per turn.

    Memory only: the supervisor publishes on the queued main_llm_call_state frame.
    Both stamps use the same clock. Unobserved admission or dispatch produces no
    row; an unfinished turn remains unobserved rather than becoming zero latency.
    """
    global _direct, _dispatch_pending
    if dispatched_at < accepted_at:
        return
    with _lock:
        _direct = {"task_id": task_id, "attempt_id": attempt_id, "accepted_at": accepted_at,
                   "dispatched_at": dispatched_at, "seconds": dispatched_at - accepted_at}
        _dispatch_pending = True


def publish_pending_dispatch() -> None:
    """Supervisor-side flush, even if the observed turn has already left the registry.

    Repeated frames and later rounds do no I/O without a new first-dispatch fact.
    Only the latest observation is retained; the lock never spans sampling or I/O.
    """
    global _dispatch_pending
    with _lock:
        if not _dispatch_pending:
            return
        _dispatch_pending = False
    _publish_after()


def projection(*, drive_root=None, state: dict | None = None) -> dict | None:
    """Addressed read for panel and mind; never sample, write, fetch or message."""
    from ouroboros.config import DATA_DIR
    from ouroboros.gateway.update_progress import STATE_KEY

    if state is None:
        state = read_json_dict(pathlib.Path(drive_root or DATA_DIR) / "state" / "state.json") or {}
    record = state.get(STATE_KEY) or {}
    facts = record.get("generation_facts")
    if not isinstance(facts, dict):
        return None
    return {"operation_id": record.get("operation_id"), "outcome": record.get("outcome"),
            "base_sha": record.get("base_sha"), "target_sha": record.get("target_sha"), **copy.deepcopy(facts)}
