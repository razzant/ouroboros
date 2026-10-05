"""In-process read acceleration for the usage ledger, beside the substrate.

The display memo/render cache permits explicitly stale presentation. The
separate existing writer view is generation-bound and strict: complete records,
last attempt rows, validation state/late-receipt rights and reversible exact cash
by root and billing group advance together under the monetary lock. Ordinary cold/replaced views
prepare outside the lock and reconcile their suffix after identity/CAS proof.
Only exceptional corruption repair uses the authoritative locked full replay.
No reader opens the archive: bindings come from the live rows alone.
Display memos capture row references; public records detach, writers borrow.

``usage_accounting`` re-binds every name here, and the implementation resolves
the substrate (``_locked``, ``_read_records_locked``, ...) through the
``usage_accounting`` namespace at call time, so the historical monkeypatch
sites (``ua._locked``, ``ua._read_records_locked``) keep governing these reads
exactly as when the code was inline.
"""
from __future__ import annotations

import collections
import contextlib
import heapq
import os
import copy
import logging
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

from ouroboros.runtime_limits import (
    USAGE_DISPLAY_LOCK_TIMEOUT_SEC,
    USAGE_DISPLAY_REVALIDATE_AFTER_SEC,
)
from ouroboros.usage_ledger import QUARANTINE_REL, LedgerResumeState, UsageLockUnavailable, is_abandoned_settlement
from ouroboros._usage_money import billing_group_key, monetary_scope_key, ZERO_CASH, cash_contribution, change_cash, render_cash, exceeds_limit
from ouroboros._usage_rows import BindingIndex

log = logging.getLogger(__name__)


def _ua():
    """The accounting namespace, resolved lazily (import cycle + test pins)."""
    from ouroboros import usage_accounting

    return usage_accounting


@dataclass
class _LedgerRowsMemo:
    """One captured writer generation and its outside-lock display renders.

    Rows are private immutable-by-ownership references: writers replace rows,
    never edit them. The tuple and scalar resume fingerprint detach this capture
    from the writer's mutable indexes. Public consumers receive deep copies.
    """

    resume: LedgerResumeState
    final_rows: tuple
    source: object
    generation: int = 0
    renders: Dict[Tuple[Any, ...], Dict[str, Any]] = field(default_factory=dict)


# Only presentation snapshots/renders live here, never a second replay cache.
_ROWS_MEMO: Dict[str, _LedgerRowsMemo] = {}
_ROWS_MEMO_LOCK = threading.Lock()
_STALE_BACKOFF: Dict[str, float] = {}


def _stale_memo_rows(key: str):
    with _ROWS_MEMO_LOCK:
        memo = _ROWS_MEMO.get(key)
        if memo is None:
            return None
        return memo.final_rows, memo.resume.st_ino != -2, memo, memo.generation


def _memoized_final_rows(root: pathlib.Path, *, allow_stale: bool = False):
    """Capture strict rows from the ONE prepared source used by writers.

    Cold/replaced history parses outside the money lock; inside it we validate
    the generation, reconcile only the suffix and capture row references.
    Rendering/folding happens after release. Internal rows are read-only;
    ``read_usage_records`` is the detached public snapshot seam.

    Display-only ``allow_stale`` retains the short acquisition/backoff and last
    validated snapshot. A cold contended read stays unavailable, never zero.
    Strict reads and admission never use the stale snapshot. Platform refusal
    is not contention and must propagate even when a snapshot exists.
    """
    ua = _ua()
    key = str(pathlib.Path(root).resolve(strict=False))
    if allow_stale:
        with _ROWS_MEMO_LOCK:
            backing_off = time.monotonic() < _STALE_BACKOFF.get(key, 0.0)
        stale = _stale_memo_rows(key) if backing_off else None
        if stale is not None:
            return stale
    acquisition = (lambda path: ua._locked(path, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC)) if allow_stale else None
    try:
        with _writer_locked(root, acquisition=acquisition) as view:
            resume = view.resume
            # Identity distinguishes rebuilt views even if their stat fingerprint
            # matches. Never retain resume.states/late_receipt_ids: they mutate.
            source = (view.identity, resume.st_ino, resume.st_dev, resume.size,
                      resume.st_mtime_ns, resume.row_count)
            with _ROWS_MEMO_LOCK:
                memo = _ROWS_MEMO.get(key)
                if memo is None or memo.source != source or resume.st_ino == -2:
                    memo = _LedgerRowsMemo(
                        LedgerResumeState(*source[1:]), tuple(view.finals.values()),
                        source, (memo.generation + 1) if memo else 0,
                    )
                    _ROWS_MEMO[key] = memo
                _STALE_BACKOFF.pop(key, None)
                return memo.final_rows, resume.st_ino != -2, memo, memo.generation
    except UsageLockUnavailable as exc:
        if not allow_stale or exc.reason != "contention":
            raise
        with _ROWS_MEMO_LOCK:
            _STALE_BACKOFF[key] = time.monotonic() + USAGE_DISPLAY_REVALIDATE_AFTER_SEC
        stale = _stale_memo_rows(key)
        if stale is None:
            raise
        return stale


def read_usage_records(root: pathlib.Path, *, final_only: bool = False) -> list:
    """Detached strict snapshot for audit/custody consumers and full readers.

    Row references are captured under lock; deep copy runs after release. All
    nested fields belong to this caller, and later appends/replacements cannot
    change its snapshot. Import/compaction that already own the lock retain the
    compatible locked reader below; do not recursively acquire the lock here.
    """
    root = _ua()._drive_root(root)
    with _writer_locked(root) as view:
        rows = list(view.finals.values()) if final_only else list(view.records)
    return copy.deepcopy(rows)


def _render_cached(
    root: pathlib.Path,
    cache_key: Tuple[Any, ...],
    render: Callable[[list, bool], Dict[str, Any]],
    *,
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
    gateway handlers) mutate nested buckets in place. ``allow_stale`` forwards
    the display-reader contract of ``_memoized_final_rows`` unchanged."""
    rows, cacheable, memo, generation = _memoized_final_rows(root, allow_stale=allow_stale)
    integrity_degraded = (root / QUARANTINE_REL).is_file()
    full_key = (*cache_key, integrity_degraded)
    if cacheable:
        with _ROWS_MEMO_LOCK:
            cached = memo.renders.get(full_key) if memo.generation == generation else None
        if cached is not None:
            return copy.deepcopy(cached)
    result = render(rows, integrity_degraded)
    if cacheable:
        key = str(pathlib.Path(root).resolve(strict=False))
        # Copy before taking the memo lock: a strict reader takes it while
        # holding money, so a rich render copy must not indirectly hold money.
        frozen = copy.deepcopy(result)
        with _ROWS_MEMO_LOCK:
            if _ROWS_MEMO.get(key) is memo and memo.generation == generation:
                memo.renders[full_key] = frozen
    return copy.deepcopy(result)


@dataclass(eq=False)
class _LedgerWriterView:
    """One private, generation-bound validated view, borrowed only under lock.

    Records/finals are retained for full-record consumers; cash is derived only
    from those final rows. No display snapshot can install this view. Public
    readers receive a list snapshot and never receive its mutable resume state.
    """

    resume: LedgerResumeState
    records: list
    identity: object = field(default_factory=object)
    finals: dict = field(default_factory=dict)
    cash: tuple = ZERO_CASH
    roots: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    bindings: BindingIndex = field(default_factory=BindingIndex)
    fold_times: list = field(default_factory=list)

    def fold(self, rows: list) -> None:
        from ouroboros.usage_compaction import fold_eligible_at

        for row in rows:
            identity = str(row["attempt_id"])
            previous = self.finals.get(identity)
            old = cash_contribution(previous) if previous else ZERO_CASH
            new = cash_contribution(row)
            self.cash = change_cash(self.cash, old, new)
            if previous:
                root = monetary_scope_key(previous)
                self.roots[root] = change_cash(self.roots[root], old)
                group = billing_group_key(previous)
                self.groups[group] = change_cash(self.groups[group], old)
            root = monetary_scope_key(row)
            self.roots[root] = change_cash(self.roots.get(root, ZERO_CASH), new=new)
            group = billing_group_key(row)
            self.groups[group] = change_cash(self.groups.get(group, ZERO_CASH), new=new)
            self.bindings.fold(row)
            self.finals[identity] = row
            eligible = fold_eligible_at(row)
            if eligible is not None:
                heapq.heappush(self.fold_times, (eligible, int(row["seq"]), identity))

    def has_foldable_attempt(self) -> bool:
        from ouroboros.usage_compaction import _fold_clock

        while self.fold_times:
            eligible, sequence, identity = self.fold_times[0]
            if self.finals[identity]["seq"] == sequence:
                return eligible <= _fold_clock()
            heapq.heappop(self.fold_times)  # superseded late receipt, once per update
        return False

    def totals(self, root_task_id=None, billing_group_id=None):
        if root_task_id is not None and billing_group_id is not None:
            raise ValueError("select one monetary axis")
        if billing_group_id is not None:
            return self.groups.get(billing_group_id, ZERO_CASH)
        return self.cash if root_task_id is None else self.roots.get(root_task_id, ZERO_CASH)

    def summary(self, root_task_id: str | None = None, *, billing_group_id=None) -> dict:
        return render_cash(self.totals(root_task_id, billing_group_id))

    def exceeds_limit(self, limit, bound=None, *, root_task_id=None, billing_group_id=None, dispatch=False):
        return exceeds_limit(self.totals(root_task_id, billing_group_id), limit, bound, dispatch=dispatch)

    def append(self, root: pathlib.Path, rows: list) -> list:
        ua = _ua()
        key = str(root.resolve(strict=False))
        try:
            appended = ua._append_rows_locked(root, self.records, rows, resume=self.resume)
            # Publication follows durable append/fsync. Any uncertainty invalidates
            # the whole view, including partially advanced arithmetic.
            self.fold(appended)
            self.records.extend(appended)
            for row in appended:
                identity = str(row["attempt_id"])
                self.resume.states[identity] = str(row["state"])
                if str(row.get("kind") or "attempt") == "attempt" and (
                    row["state"] == "unresolved" or is_abandoned_settlement(row)
                ):
                    self.resume.late_receipt_ids.add(identity)
                else:
                    self.resume.late_receipt_ids.discard(identity)
            stat = (root / ua.LEDGER_REL).stat()
            self.resume = LedgerResumeState(stat.st_ino, stat.st_dev, stat.st_size, stat.st_mtime_ns,
                                            len(self.records), self.resume.states, self.resume.late_receipt_ids)
            return appended
        except BaseException:
            with _LEDGER_READ_CACHE_LOCK:
                _LEDGER_READ_CACHE.pop(key, None)
            raise


_LEDGER_READ_CACHE: collections.OrderedDict[str, _LedgerWriterView] = collections.OrderedDict()
_LEDGER_READ_CACHE_LOCK = threading.Lock()
_LEDGER_READ_CACHE_MAX_ROOTS = 8


def _ledger_cache_put(key: str, value: _LedgerWriterView) -> None:
    with _LEDGER_READ_CACHE_LOCK:
        _LEDGER_READ_CACHE[key] = value
        _LEDGER_READ_CACHE.move_to_end(key)
        while len(_LEDGER_READ_CACHE) > _LEDGER_READ_CACHE_MAX_ROOTS:
            _LEDGER_READ_CACHE.popitem(last=False)


def _seed_writer(records: list, resume: LedgerResumeState) -> _LedgerWriterView:
    view = _LedgerWriterView(resume, records)
    view.fold(records)
    return view


_PREPARATION_CHANGED = object()


def _prepare_writer(root: pathlib.Path):
    """Read/validate a captured newline-aligned extent without the money lock.

    A preparation is unpublished and has no quarantine authority. Atomic
    replacement/shrink/same-size rewrite is proved again under lock. History
    is append-only within one inode; any other rewrite must atomically replace
    the file, or a same-inode rewrite that grows it can go undetected.
    """
    from ouroboros.usage_ledger import _decode_record

    ua = _ua()
    try:
        with open(root / ua.LEDGER_REL, "rb") as handle:
            stat = os.fstat(handle.fileno())
            records = []
            consumed = 0
            while consumed < stat.st_size:
                chunk = handle.readline(stat.st_size - consumed)
                if not chunk.endswith(b"\n"):
                    break
                consumed += len(chunk)
                for line in chunk.splitlines(keepends=True):
                    row = _decode_record(line)
                    if row is not None:
                        records.append(row)
            after = os.fstat(handle.fileno())
            if after.st_size < stat.st_size or (after.st_size == stat.st_size
                                               and after.st_mtime_ns != stat.st_mtime_ns):
                return _PREPARATION_CHANGED
        states, late_ids = {}, set()
        ua._validate_records(records, states=states, late_receipt_ids=late_ids)
        resume = LedgerResumeState(stat.st_ino, stat.st_dev, consumed, stat.st_mtime_ns,
                                   len(records), states, late_ids)
        return _seed_writer(records, resume)
    except FileNotFoundError:
        return _seed_writer([], LedgerResumeState(-1, -1, 0, -1, 0))
    except (OSError, ValueError, ua.UsageLedgerCorrupt):
        # Only the locked full reader may judge and quarantine a damaged tail.
        return None


def _writer_generation(root: pathlib.Path) -> tuple:
    try:
        stat = (root / _ua().LEDGER_REL).stat()
    except FileNotFoundError:
        return (-1, -1, 0, -1)
    return stat.st_ino, stat.st_dev, stat.st_size, stat.st_mtime_ns


def _writer_generation_matches(root: pathlib.Path, view: _LedgerWriterView) -> bool:
    inode, device, size, mtime = _writer_generation(root)
    resume = view.resume
    return ((inode, device) == (resume.st_ino, resume.st_dev)
            and size >= resume.size and (size != resume.size or mtime == resume.st_mtime_ns))


def _advance_writer(root: pathlib.Path, view: _LedgerWriterView) -> bool:
    ua = _ua()
    delta = ua._read_new_records_locked(root, view.resume, private=True)
    if delta is None:
        return False
    rows, resume = delta
    view.fold(rows)
    view.records.extend(rows)
    view.resume = resume
    return True


@contextlib.contextmanager
def _writer_locked(root: pathlib.Path, *, before_read=None, acquisition=None):
    """Prepare outside, prove and reconcile inside, then lend one private view.

    Replacement (including a compaction in this acquisition) releases the lock
    and prepares again. The cache object is a CAS token: an older preparation
    never overwrites a newer installed view. The transaction body is yielded
    once and is NEVER retried, even when append or fsync fails.
    """
    ua = _ua()
    key = str(root.resolve(strict=False))
    while True:
        with _LEDGER_READ_CACHE_LOCK:
            expected = _LEDGER_READ_CACHE.get(key)
        prepared = None
        needs_preparation = expected is None or not _writer_generation_matches(root, expected)
        if needs_preparation:
            prepared = _prepare_writer(root)
            if prepared is _PREPARATION_CHANGED:
                continue
        with (acquisition(root) if acquisition else ua._locked(root)) as heartbeat:
            with _LEDGER_READ_CACHE_LOCK:
                view = _LEDGER_READ_CACHE.get(key)
            try:
                if view is not None and _writer_generation_matches(root, view):
                    if not _advance_writer(root, view):
                        # Only an invalid suffix requires locked quarantine.
                        records = ua._read_records_locked(root)
                        view = _seed_writer(records, ua._ledger_resume_state(root, records))
                elif view is expected and prepared is not None and _writer_generation_matches(root, prepared):
                    if not _advance_writer(root, prepared):
                        records = ua._read_records_locked(root)
                        prepared = _seed_writer(records, ua._ledger_resume_state(root, records))
                    view = prepared
                elif view is expected and needs_preparation and prepared is None:
                    records = ua._read_records_locked(root)
                    view = _seed_writer(records, ua._ledger_resume_state(root, records))
                else:
                    continue
                _ledger_cache_put(key, view)
                if before_read:
                    generation = _writer_generation(root)
                    before_read(heartbeat, view)
                    if _writer_generation(root) != generation:
                        continue  # committed maintenance: reprepare outside lock
                yield view
                return
            except BaseException:
                with _LEDGER_READ_CACHE_LOCK:
                    _LEDGER_READ_CACHE.pop(key, None)
                raise


def _read_records_locked_cached(root: pathlib.Path) -> list:
    """Compatible detached full-record snapshot for already-locked imports.

    These callers already hold the lock. Monetary hot paths instead borrow
    ``_writer_locked`` so neither cold preparation nor list copies run there.
    """
    ua = _ua()
    key = str(root.resolve(strict=False))
    with _LEDGER_READ_CACHE_LOCK:
        view = _LEDGER_READ_CACHE.get(key)
    try:
        if view is None or not _advance_writer(root, view):
            records = ua._read_records_locked(root)
            view = _seed_writer(records, ua._ledger_resume_state(root, records))
        _ledger_cache_put(key, view)
        return copy.deepcopy(view.records)
    except BaseException:
        with _LEDGER_READ_CACHE_LOCK:
            _LEDGER_READ_CACHE.pop(key, None)
        raise
