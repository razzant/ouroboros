"""One process's bounded, best-effort dialogue retry buffer; never a required outbox.

The existing delivery duty drains the server; workers drain before their next
append and at exit. Rows are immutable JSON, oldest first, capped in rows AND
encoded bytes. Crash, overflow and an unavailable store at exit can lose rows.
Retries use the original content address and emit no second live frame.
"""
from __future__ import annotations

import atexit
import json
import logging
import pathlib
import threading
from collections import OrderedDict

from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

log = logging.getLogger(__name__)
MAX_ROWS = 2048
MAX_BYTES = 4 * 1024 * 1024
_rows: OrderedDict = OrderedDict()
_lock = threading.Lock()
_draining = threading.Lock()
_bytes = 0
_counts = dict(appended=0, retried_ok=0, evicted=0, lost_at_exit=0)
_eviction_burst = False
_wake = None
_requested = threading.Event()


def bind_wake(wake) -> None:
    """Bind the server's existing duty; no scheduler or thread belongs here."""
    global _wake
    _wake = wake
    if snapshot()["rows"] or _requested.is_set():
        wake()


def request_drain() -> None:
    """Wake the existing delivery duty, including a required outbox continuation."""
    _requested.set()
    if _wake is not None:
        _wake()


def retain(root, stream, row, *, feed=False, **shown) -> None:
    """Take immutable bytes without waiting for IO; disclose each overflow burst. ``shown`` keeps the
    append's appearance facts (``dialogue_store.append``: ``also_feed``, ``room_revisions``,
    ``appearances``) for the retry, so a retried row is unread and found where it is shown."""
    from ouroboros.chat_chain import row_address

    global _bytes, _eviction_burst
    address = row_address(row)
    key = (str(pathlib.Path(root).resolve()), stream, address["chat_id"], address["ts"], address["row_sha256"])
    payload = json.dumps([key[0], stream, row, bool(feed), shown], ensure_ascii=False).encode("utf-8")
    announce = False
    with _lock:
        if key in _rows:
            return
        _rows[key] = payload
        _bytes += len(payload)
        _counts["appended"] += 1
        removed = 0
        while len(_rows) > MAX_ROWS or _bytes > MAX_BYTES:
            # Progress narration goes first; chat rows (the owner's own words among them) wait longest.
            victim = next((name for name in _rows if name[1] == "progress"), None) or next(iter(_rows))
            _bytes -= len(_rows.pop(victim))
            removed += 1
        if removed:
            _counts["evicted"] += removed
            announce = not _eviction_burst
            _eviction_burst = True
    if announce:
        log.info("Dialogue retry capacity exceeded; oldest rows evicted, progress first (counts in dialogue_retry.snapshot)")
    request_drain()


def snapshot() -> dict:
    with _lock:
        return {**_counts, "rows": len(_rows), "bytes": _bytes,
                "max_rows": MAX_ROWS, "max_bytes": MAX_BYTES}


def drain(*, wait_sec=ACQUISITION_SLICE_SEC) -> int:
    """Attempt the finite retained snapshot once, stopping at the first failed write.

    The drain lock is nonblocking; producers never wait behind a store writer.
    An evicted row already being committed can still land (eviction is counted).
    """
    from ouroboros import dialogue_store as ds, store

    global _bytes, _eviction_burst
    if not _draining.acquire(blocking=False):
        return 0
    done = 0
    try:
        with _lock:
            batch = list(_rows)
        for key in batch:
            with _lock:
                payload = _rows.get(key)
                if payload is None:
                    continue
            root, stream, row, feed, shown = json.loads(payload)
            try:
                with store.acquisition_budget(wait_sec):
                    ds.append(root, stream, row, feed=feed, idempotent=True, emit_live=False, **shown)
                    if stream == "chat":
                        from ouroboros.notice_receipts import record
                        record(root, row)
            except (store.StoreError, OSError, ValueError):
                break
            with _lock:
                old = _rows.pop(key, None)
                if old is not None:
                    _bytes -= len(old)
                _counts["retried_ok"] += 1
                if not _rows:
                    _eviction_burst = False
            done += 1
    finally:
        _draining.release()
    return done


def before_append() -> None:
    """Workers have no delivery duty: try prior rows without delaying this live row."""
    from ouroboros.utils import in_worker_process
    if in_worker_process():
        drain(wait_sec=0)


def finish() -> None:
    """One bounded exit attempt; count and disclose rows the process cannot keep."""
    global _bytes
    drain()
    with _lock:
        lost = len(_rows)
        _counts["lost_at_exit"] += lost
        _rows.clear()
        _bytes = 0
    if lost:
        log.info("Dialogue retry process exit lost %d retained rows", lost)


atexit.register(finish)
# multiprocessing workers run their own finalizers even when bypassing atexit.
from multiprocessing.util import Finalize
_finalizer = Finalize(None, finish, exitpriority=10)
