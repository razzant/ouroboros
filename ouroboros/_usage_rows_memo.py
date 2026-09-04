"""In-process read acceleration for the usage ledger, beside the substrate.

Extracted from ``usage_accounting.py`` at the perf2 P1 render-cache round for
the same reason ``_usage_rows.py`` exists: the accounting module sits at the
hard module-size gate and this layer is a self-contained seam. It carries the
ledger's two in-process caches — the validated-rows memo + render cache for
display projections, and the in-lock warm read cache the monetary write paths
use (razzant/ouroboros#129) — and it deliberately lives beside, not inside,
``usage_ledger.py``: the substrate stays cache-ignorant. In every case the
full ``_read_records_locked`` replay remains the authority and the sole owner
of quarantine; a cache can only ever change the COST of a read, never its
result, because ``_read_new_records_locked`` re-stats the file under the held
lock and refuses to resume on any doubt.

``usage_accounting`` re-binds every name here, and the implementation resolves
the substrate (``_locked``, ``_read_records_locked``, ...) through the
``usage_accounting`` namespace at call time, so the historical monkeypatch
sites (``ua._locked``, ``ua._read_records_locked``) keep governing these reads
exactly as when the code was inline.
"""
from __future__ import annotations

import collections
import copy
import json
import logging
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros.usage_ledger import (
    QUARANTINE_REL,
    LedgerResumeState,
    UsageLockUnavailable,
)

log = logging.getLogger(__name__)


def _ua():
    """The accounting namespace, resolved lazily (import cycle + test pins)."""
    from ouroboros import usage_accounting

    return usage_accounting


@dataclass
class _LedgerRowsMemo:
    """In-process cache of one drive root's per-attempt FINAL rows.

    Holds only the ``_final_rows`` dict (one row per attempt, first-occurrence
    order) plus the resume fingerprint — O(final rows), not O(ledger rows);
    superseded transition rows are not retained.

    ``renders`` is the fingerprint-keyed cache of finished display renders
    (``usage_projection``/``usage_breakdown`` bodies) computed over these rows:
    valid exactly while the rows are, so it is cleared on refold and on every
    non-empty advance, never by TTL. ``generation`` increments on those same
    two events and guards the clear-then-publish race (see ``_render_cached``).
    """

    resume: LedgerResumeState
    final_rows: Dict[str, Dict[str, Any]]
    generation: int = 0
    renders: Dict[Tuple[Any, ...], Dict[str, Any]] = field(default_factory=dict)


# Read-side memo per RESOLVED drive root. Populated and advanced only under the
# cross-process ledger lock; the module lock guards the dict itself. Write paths
# (reserve/_transition/settle/import) never touch it — they read through their
# own in-lock cache below (full ordered records, which seq assignment and
# whole-history append validation need; this memo keeps only final rows), and
# the stat + seq-continuity check on the next read is what makes a stale memo
# impossible to serve, so correctness never depends on any writer remembering
# to invalidate.
_ROWS_MEMO: Dict[str, _LedgerRowsMemo] = {}
_ROWS_MEMO_LOCK = threading.Lock()

# Stale-while-revalidate backoff for display readers (``allow_stale=True``):
# after a contended lock attempt, further display reads serve the memo
# lock-free for this long instead of paying ``lock_timeout_sec`` per call.
# The bound keeps a convoyed supervisor loop / gateway event loop at ~zero
# lock-wait per second while still revalidating roughly once a second.
_STALE_REVALIDATE_AFTER_SEC = 1.0
_STALE_BACKOFF: Dict[str, float] = {}


def _stale_memo_rows(key: str) -> "Tuple[list, bool, _LedgerRowsMemo, int] | None":
    """The last validated snapshot for ``key``, lock-free, or None when cold."""
    with _ROWS_MEMO_LOCK:
        memo = _ROWS_MEMO.get(key)
        if memo is None:
            return None
        return (
            list(memo.final_rows.values()),
            memo.resume.st_ino != -2,
            memo,
            memo.generation,
        )


def _memoized_final_rows(
    root: pathlib.Path,
    *,
    lock_timeout_sec: float = 45.0,
    allow_stale: bool = False,
) -> Tuple[list, bool, "_LedgerRowsMemo", int]:
    """Validated final rows for display projections, resumed incrementally.

    Cold (or whenever the resume fingerprint is rejected — file replacement,
    size shrink, same-size rewrite, seq discontinuity, a non-row-aligned tail,
    structural corruption) this is one full ``_read_records_locked`` replay,
    which owns quarantine. Warm, it parses only the bytes appended since the
    previous read. Locks and substrate reads resolve through the
    ``usage_accounting`` namespace at call time (tests monkeypatch those
    names). Returned row dicts are shared read-only snapshots; row ORDER
    matches a from-scratch ``_final_rows`` exactly (first-occurrence order,
    updates in place), so aggregation over them is bit-identical to a fresh
    replay.

    Returns ``(rows, cacheable, memo, generation)`` — the render-cache
    transport: ``cacheable`` is False for the deliberately NON-RESUMABLE
    crash-tail fingerprint (``st_ino == -2``), whose every read stays a full
    replay, so caching a render of it would hide exactly the reads that must
    keep re-checking the torn tail. ``memo``/``generation`` let
    ``_render_cached`` publish a render computed OUTSIDE the lock only if the
    rows have not moved since.

    ``allow_stale`` is the display-reader contract: the lock is attempted with
    ``lock_timeout_sec``, and a contended lock serves the last validated
    snapshot (the memo's rows are immutable and only ever published under
    ``_ROWS_MEMO_LOCK``) instead of raising — a display projection may lag the
    ledger by seconds, but must never stall a concurrency-critical caller.
    A cold memo (nothing validated yet) still raises ``UsageLockUnavailable``,
    so callers degrade to their "unavailable" branch exactly once per convoy.
    While the backoff is active the lock is not attempted at all.
    """
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))
    if allow_stale:
        with _ROWS_MEMO_LOCK:
            backoff_until = _STALE_BACKOFF.get(key, 0.0)
        if time.monotonic() < backoff_until:
            stale = _stale_memo_rows(key)
            if stale is not None:
                return stale
    try:
        with ua._locked(root, timeout_sec=lock_timeout_sec):
            with _ROWS_MEMO_LOCK:
                memo = _ROWS_MEMO.get(key)
            advanced = ua._read_new_records_locked(root, memo.resume) if memo is not None else None
            if advanced is None:
                records = ua._read_records_locked(root)
                memo = _LedgerRowsMemo(
                    resume=ua._ledger_resume_state(root, records),
                    final_rows=ua._final_rows(records),
                    generation=(memo.generation + 1) if memo is not None else 0,
                )
            else:
                new_records, new_resume = advanced
                if new_records:
                    for row in new_records:
                        memo.final_rows[str(row["attempt_id"])] = row
                    # The generation bump and the renders clear MUST happen under
                    # _ROWS_MEMO_LOCK: the publisher in _render_cached checks the
                    # generation and writes under that lock only (it never holds
                    # the ledger lock), so without it the check-then-publish pair
                    # could interleave with this clear — a stale render published
                    # right after the clear would then serve pre-append data to
                    # every warm reader until the next append. Lock order stays
                    # "ledger lock → memo lock" (same as above/below); the
                    # publisher takes the memo lock alone, so no deadlock. The
                    # refold branch needs no such section: it swaps in a NEW memo
                    # object and the publisher's `is memo` identity check already
                    # rejects publications against a replaced object.
                    with _ROWS_MEMO_LOCK:
                        memo.generation += 1
                        memo.renders.clear()
                memo.resume = new_resume
            with _ROWS_MEMO_LOCK:
                _ROWS_MEMO[key] = memo
                if allow_stale:
                    _STALE_BACKOFF.pop(key, None)
            cacheable = memo.resume.st_ino != -2
            return list(memo.final_rows.values()), cacheable, memo, memo.generation
    except UsageLockUnavailable:
        if not allow_stale:
            raise
        with _ROWS_MEMO_LOCK:
            _STALE_BACKOFF[key] = time.monotonic() + _STALE_REVALIDATE_AFTER_SEC
        stale = _stale_memo_rows(key)
        if stale is None:
            raise
        return stale


def _render_cached(
    root: pathlib.Path,
    cache_key: Tuple[Any, ...],
    render: Callable[[list, bool], Dict[str, Any]],
    *,
    lock_timeout_sec: float = 45.0,
    allow_stale: bool = False,
) -> Dict[str, Any]:
    """Serve one display render through the memo's fingerprint-keyed cache.

    The cache lives INSIDE the memo, so its lifetime is exactly the rows':
    refold and non-empty advance both replace/clear ``renders`` under the
    ledger lock. The quarantine stat happens HERE — outside the memo but after
    the row read, because that read itself may quarantine a torn tail — and the
    resulting bool joins the cache key, so an integrity change alone can never
    serve a stale render. The render itself runs OUTSIDE any lock; publication
    happens under ``_ROWS_MEMO_LOCK`` and only when the memo object and its
    generation are unchanged since the rows were read — a concurrent append
    between read and publish means the render is returned to this caller but
    never cached. Both directions hand out deep copies: the cached object is
    shared between requests, and callers (``_with_limit``/``_with_integrity``,
    gateway handlers) mutate nested buckets in place.

    ``lock_timeout_sec``/``allow_stale`` forward to ``_memoized_final_rows``:
    display readers ride out a contended ledger lock on the last validated
    snapshot instead of stalling their caller."""
    rows, cacheable, memo, generation = _memoized_final_rows(
        root, lock_timeout_sec=lock_timeout_sec, allow_stale=allow_stale
    )
    integrity_degraded = (root / QUARANTINE_REL).is_file()
    full_key = (*cache_key, integrity_degraded)
    if cacheable:
        with _ROWS_MEMO_LOCK:
            if memo.generation == generation:
                cached = memo.renders.get(full_key)
                if cached is not None:
                    return copy.deepcopy(cached)
    result = render(rows, integrity_degraded)
    if cacheable:
        key = str(pathlib.Path(root).resolve(strict=False))
        with _ROWS_MEMO_LOCK:
            if _ROWS_MEMO.get(key) is memo and memo.generation == generation:
                memo.renders[full_key] = copy.deepcopy(result)
    return result


# razzant/ouroboros#129: the in-lock write paths (reserve/settle/_transition/
# release/legacy-import) each did a full parse+validate of the whole ledger
# under the 45s monetary flock, and the file grows unboundedly. This is their
# per-process warm cache of the last validated read per drive root: the next
# in-lock read parses only the bytes appended since. It is distinct from
# ``_ROWS_MEMO`` because writers need the FULL ordered records list (seq
# assignment + whole-history append validation), not just final rows. Rows are
# shared read-only snapshots, same as the memo's.
_LEDGER_READ_CACHE: "collections.OrderedDict[str, Tuple[LedgerResumeState, list]]" = (
    collections.OrderedDict()
)
_LEDGER_READ_CACHE_LOCK = threading.Lock()
_LEDGER_READ_CACHE_MAX_ROOTS = 8
# Unlocked warm-up attempts that found an unusable file (torn/corrupt) are not
# repeated on every lock entry; the locked full read owns quarantine.
_WARM_FAILED_AT: Dict[str, float] = {}
_WARM_RETRY_SEC = 30.0


def _ledger_cache_put(key: str, value: "Tuple[LedgerResumeState, list]") -> None:
    with _LEDGER_READ_CACHE_LOCK:
        _LEDGER_READ_CACHE[key] = value
        _LEDGER_READ_CACHE.move_to_end(key)
        while len(_LEDGER_READ_CACHE) > _LEDGER_READ_CACHE_MAX_ROOTS:
            _LEDGER_READ_CACHE.popitem(last=False)


def _warm_ledger_read_cache(root: pathlib.Path) -> bool:
    """Seed this process's ledger read cache WITHOUT the monetary lock.

    A process's first in-lock read parses and validates the whole ledger under
    the lock (0.7-1 s at 45K rows, linear in size). When a deadline wave frees
    ~48 lanes at once, 48 fresh workers do that cold read back to back under
    the same lock — ~50 s of serialized hold — and every reserve behind them
    times out (CyberGym r8, 2026-09-04 08:27-08:31: 15 tasks died as
    task_exception: UsageLockUnavailable within minutes of admission).

    The ledger is append-only under the lock, so a validated snapshot of a
    row-aligned prefix taken without the lock is exactly what the incremental
    resume seam expects: the in-lock read then parses only the bytes appended
    since. Anything doubtful (torn tail, parse or validation failure, a
    concurrent rewrite) leaves the cache untouched and the locked full read —
    which OWNS quarantine — runs as before. Returns True when a snapshot was
    seeded. Never mutates the ledger.
    """
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))
    now = time.monotonic()
    with _LEDGER_READ_CACHE_LOCK:
        if key in _LEDGER_READ_CACHE:
            return False
        failed_at = _WARM_FAILED_AT.get(key)
        if failed_at is not None and now - failed_at < _WARM_RETRY_SEC:
            return False
    path = pathlib.Path(root) / ua.LEDGER_REL
    try:
        with open(path, "rb") as handle:
            data = handle.read()
            stat = os.fstat(handle.fileno())
    except OSError:
        return False
    boundary = data.rfind(b"\n") + 1
    if boundary <= 0:
        return False

    def _give_up() -> bool:
        with _LEDGER_READ_CACHE_LOCK:
            _WARM_FAILED_AT[key] = time.monotonic()
        return False

    records: list = []
    for chunk in data[:boundary].splitlines():
        raw = chunk.rstrip(b"\r")
        if not raw:
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _give_up()
        if not isinstance(row, dict):
            return _give_up()
        records.append(row)
    states: Dict[str, str] = {}
    try:
        ua._validate_records(records, states=states)
    except ua.UsageLedgerCorrupt:
        return _give_up()
    # The mtime fingerprint only matters when the size is unchanged; if a writer
    # appended between our read and the stat, the sizes differ and the in-lock
    # tail read takes over from ``boundary`` regardless.
    mtime_ns = stat.st_mtime_ns if boundary == stat.st_size else -1
    resume = ua.LedgerResumeState(
        stat.st_ino, stat.st_dev, boundary, mtime_ns, len(records), states,
    )
    with _LEDGER_READ_CACHE_LOCK:
        if key in _LEDGER_READ_CACHE:
            return False
        _LEDGER_READ_CACHE[key] = (resume, records)
        _LEDGER_READ_CACHE.move_to_end(key)
        while len(_LEDGER_READ_CACHE) > _LEDGER_READ_CACHE_MAX_ROOTS:
            _LEDGER_READ_CACHE.popitem(last=False)
    return True


def _cached_attempt_states(root: pathlib.Path, row_count: int) -> Optional[Dict[str, str]]:
    """Per-attempt last-state map of the cached validated read, if it covers
    exactly ``row_count`` rows; a copy the caller may mutate. None = not cached
    for that extent (caller derives the map from its records)."""
    key = str(pathlib.Path(root).resolve(strict=False))
    with _LEDGER_READ_CACHE_LOCK:
        cached = _LEDGER_READ_CACHE.get(key)
    if cached is None:
        return None
    resume, rows = cached
    if int(resume.row_count) != int(row_count) or len(rows) != int(row_count):
        return None
    return dict(resume.states)


class _FinalsView:
    """Final rows, their ``_summary`` and a per-root index, maintained incrementally.

    ``reserve_attempt`` folded ``_final_rows`` + ``_summary`` over the WHOLE
    cached record list under the monetary lock on every reserve, and filtered
    the finals per root task on top: O(ledger) per call. Measured on the r8
    ledger replicated to 512K rows: ~100 ms for the dict alone, and the summary
    grows with the number of attempts (~170K at the tail of a 1369-task
    campaign) — enough, at 1-2 reserves/s, to hand the lock convoy back in the
    last third of the run. This view applies only the rows appended since its
    last call: superseded final rows are retracted from the accumulator and the
    new ones added, so the summary is the same arithmetic as a replay.

    The view is keyed on the cached record list's shape — ``(first seq, last
    seq, row count)`` — and any mismatch (cache refold, quarantine, shrink,
    another root) rebuilds from scratch. It never touches the file; it is
    always called under the held ledger lock with the list the in-lock read
    just returned. Every ``_REFOLD_EVERY`` applied rows it refolds anyway, so
    float add/retract drift stays bounded, and once a subscription row is seen
    (whose window maximum is not retractable) the summary is re-folded from
    the final rows on every call.
    """

    _REFOLD_EVERY = 4096

    def __init__(self) -> None:
        self.row_count = 0
        self.first_seq: Any = None
        self.last_seq: Any = None
        self.finals: Dict[str, Dict[str, Any]] = {}
        self.by_root: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.accumulator = _ua().SummaryAccumulator()
        self.applied_since_refold = 0

    def _extends(self, records: list) -> bool:
        if not self.row_count or len(records) < self.row_count:
            return False
        try:
            return (
                records[0].get("seq") == self.first_seq
                and records[self.row_count - 1].get("seq") == self.last_seq
            )
        except (AttributeError, IndexError):
            return False

    def _apply(self, row: Dict[str, Any]) -> None:
        attempt_id = str(row.get("attempt_id") or "")
        previous = self.finals.get(attempt_id)
        if previous is not None:
            self.accumulator.add(previous, -1)
            old_root = str(previous.get("root_task_id") or "")
            if old_root:
                bucket = self.by_root.get(old_root)
                if bucket is not None:
                    bucket.pop(attempt_id, None)
        self.finals[attempt_id] = row
        self.accumulator.add(row)
        root_task_id = str(row.get("root_task_id") or "")
        if root_task_id:
            self.by_root.setdefault(root_task_id, {})[attempt_id] = row
        self.applied_since_refold += 1

    def _rebuild(self, records: list) -> None:
        self.finals = {}
        self.by_root = {}
        self.accumulator = _ua().SummaryAccumulator()
        for row in records:
            self._apply(row)
        self.applied_since_refold = 0
        self._stamp(records)

    def _stamp(self, records: list) -> None:
        self.row_count = len(records)
        self.first_seq = records[0].get("seq") if records else None
        self.last_seq = records[-1].get("seq") if records else None

    def advance(self, records: list) -> None:
        if not self._extends(records) or self.applied_since_refold >= self._REFOLD_EVERY:
            self._rebuild(records)
            return
        for row in records[self.row_count:]:
            self._apply(row)
        self._stamp(records)

    def summary(self) -> Dict[str, Any]:
        if not self.accumulator.exact_retraction:
            return _ua()._summary(list(self.finals.values()))
        return self.accumulator.render()

    def rows_for_root(self, root_task_id: str) -> list:
        """Final rows of one root task's subtree (order: first occurrence, like
        a filtered ``_final_rows``)."""
        return list((self.by_root.get(str(root_task_id or "")) or {}).values())


_FINALS_VIEWS: Dict[str, _FinalsView] = {}
_FINALS_VIEWS_LOCK = threading.Lock()


def _finals_view(root: pathlib.Path, records: list) -> _FinalsView:
    """The incrementally maintained finals view for ``records`` — the list the
    in-lock read just returned for ``root``. Call under the held ledger lock."""
    key = str(pathlib.Path(root).resolve(strict=False))
    with _FINALS_VIEWS_LOCK:
        view = _FINALS_VIEWS.get(key)
        if view is None:
            view = _FINALS_VIEWS[key] = _FinalsView()
        view.advance(records)
        return view


def _read_records_locked_cached(root: pathlib.Path) -> list:
    """``_read_records_locked`` with an incremental warm path. Call under the
    held ledger lock (same contract as ``_read_records_locked``)."""
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))  # one slot per physical root
    with _LEDGER_READ_CACHE_LOCK:
        cached = _LEDGER_READ_CACHE.get(key)
    if cached is not None:
        resume, rows = cached
        try:
            delta = ua._read_new_records_locked(root, resume)
        except Exception:  # noqa: BLE001 — any doubt = fall back to the full read
            delta = None
        if delta is not None:
            new_rows, new_resume = delta
            merged = rows if not new_rows else [*rows, *new_rows]
            _ledger_cache_put(key, (new_resume, merged))
            return list(merged)
    records = ua._read_records_locked(root)
    try:
        resume = ua._ledger_resume_state(root, records)
        _ledger_cache_put(key, (resume, list(records)))
    except Exception:  # noqa: BLE001 — caching is best-effort; correctness is the full read
        log.debug("ledger read-cache seed failed for %s", key, exc_info=True)
        with _LEDGER_READ_CACHE_LOCK:
            _LEDGER_READ_CACHE.pop(key, None)
    return records
