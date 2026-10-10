"""Process-local memo of the delegated-run custody rows (razzant/ouroboros#804).

The boot and maintenance custody reads use the addressed open set instead
(``delegate_custody_current``, ARCHITECTURE invariant 10). A custody question
asked outside that binding — ownership, timing, pending invocations, open runs,
evidence on ``delegate_wait``, ``delegate_start``, task start and completion —
is still answered from the rows ``delegate_custody.emit`` appended to the
rotated ``logs/events.jsonl`` chain. That chain is durable history and grows
without bound (hundreds of MB on a long-lived install), so re-reading it on
every such call is the "full-history scan filtered down to the answer"
DEVELOPMENT §03 forbids on an interactive path.

This module is the warm-cache half of the fix: an in-process copy of the custody rows plus
a fingerprint of the chain prefix they were read from, advanced by folding only
the bytes appended since the previous read and REFOLDED FROM SCRATCH on any
doubt. The durable rows stay the one authority (ARCHITECTURE §10, invariant 10):
nothing here is written to disk, a refold costs exactly what one
``_iter_rows`` replay costs today, and an unreadable chain bypasses the memo
for that call instead of caching an "empty because unreadable" answer. It is
deliberately NOT the durable compact projection §03 also names
(``containment_faults.jsonl`` is that shape): each process pays one cold fold.

Fingerprint rule: the memo's
segments must be the chain's first ``k`` segments by ``(st_dev, st_ino)`` in
order — a rotated live file keeps its inode under its archive name, so the
prefix it consumed stays consumed; every consumed segment has
``size >= consumed``; one whose size did not move must keep its ``st_mtime_ns``
(a same-size rewrite refolds); a segment that grew is folded from ``consumed``
onward. A torn LIVE tail (no trailing newline) waits for the next call; a torn
line inside an immutable archive can never complete, so its bytes are consumed
and counted (the ``refresh_recently_settled_terminals`` rule). The live
file's consumed prefix is additionally hashed and re-verified before every
advance (it is the only segment allowed to grow, and growth alone cannot prove
the prefix); an archive is immutable by contract, so a same-size rewrite shows
in its mtime and any growth refolds. Disclosed residual: a rewrite of an
archive that preserves both size and mtime is invisible — the log is
append-only by construction (``append_jsonl`` under the writer lock, rotation
by ``os.replace``), and tests reset the memo through ``reset_custody_memo``
(autouse fixture in ``tests/conftest.py``).

Inline request bodies (legacy ``delegate_run_start_requested`` rows, hundreds of
KB each) are not retained: the row carries a ``request_locator`` instead and
``delegate_pending.request_body`` re-reads that one line on demand, so the
memo holds a few MB of compact rows, never the legacy bodies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.utils import JsonlChainUnreadable, jsonl_archive_segments

log = logging.getLogger("ouroboros.delegate_custody")

REQUEST_LOCATOR_KEY = "request_locator"


def _custody():
    """The custody facade, resolved at call time (import cycle + test pins)."""
    from ouroboros import delegate_custody

    return delegate_custody


class _Refold(Exception):
    """A consumed segment changed in a way the prefix rule cannot advance over."""


@dataclass
class _Segment:
    st_dev: int
    st_ino: int
    consumed: int
    st_mtime_ns: int
    # SHA-256 of the consumed bytes, kept only for the chain's LAST segment (the
    # live file): the one segment that may legitimately grow, so size growth alone
    # cannot certify its consumed prefix — an in-place rewrite followed by an
    # append would otherwise fold from the old offset over stale rows. Verified
    # by re-reading the prefix (bounded by the rotation cap) before each advance.
    prefix_sha256: str = ""


@dataclass
class _ChainMemo:
    segments: List[_Segment] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    rows_view: Tuple[Dict[str, Any], ...] = ()
    generation: int = 0
    torn_archive_lines: int = 0
    # Custody-marked lines the fold could NOT parse (bounded). A START_REQUESTED
    # joined onto a torn prefix lands here; readers that must prove absence
    # consult it instead of trusting the silent skip (#1196, Astra 6fe5 #2).
    malformed_marker_lines: List[bytes] = field(default_factory=list)
    malformed_overflow: bool = False
    malformed_bytes: int = 0
    # (generation, folded state) for ``folded_state``; cloned on every return.
    state_cache: Optional[Tuple[int, Any]] = None


_MEMOS: Dict[str, _ChainMemo] = {}
_LOCKS: Dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _key(path: pathlib.Path) -> str:
    return str(pathlib.Path(path).resolve(strict=False))


def _lock_for(key: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def reset_custody_memo(drive_root: Any = None) -> None:
    """Forget one drive root's memo (or every memo): the next read refolds."""
    with _REGISTRY_LOCK:
        if drive_root is None:
            _MEMOS.clear()
            return
        _MEMOS.pop(_key(_custody().event_log_path(drive_root)), None)


def _enumerate(path: pathlib.Path) -> List[Tuple[pathlib.Path, os.stat_result, bool]]:
    """The chain as ``(segment, stat, is_live)`` in fold order; strict on archives.

    The live file is pinned by identity BEFORE the archives are listed (the
    ``jsonl_chain_handles`` rule): a rotation between the two steps renames
    that inode into the archive, where it is then excluded as the live file's
    alias instead of being counted twice or lost from both lists.
    """
    try:
        live_stat: Optional[os.stat_result] = path.stat()
    except FileNotFoundError:
        live_stat = None  # an absent live file is a positively empty tail
    live_identity = (live_stat.st_dev, live_stat.st_ino) if live_stat is not None else None
    chain: List[Tuple[pathlib.Path, os.stat_result, bool]] = []
    for segment in jsonl_archive_segments(path, strict=True):
        try:
            stat = segment.stat()
        except FileNotFoundError:
            continue  # rotated away between enumeration and stat: not part of history
        except OSError as exc:
            raise JsonlChainUnreadable(f"cannot stat {segment}: {exc}") from exc
        if (stat.st_dev, stat.st_ino) == live_identity:
            continue  # the pinned live generation under its new archive name
        chain.append((segment, stat, False))
    if live_stat is not None:
        chain.append((path, live_stat, True))
    return chain


def _open_identified(segment: pathlib.Path, stat: os.stat_result):
    """Open ``segment`` and prove the handle is the enumerated inode.

    A rotation between the stat and this open would otherwise hand the fold a
    NEW empty live file under the OLD identity, and its consumed offset would
    be applied to bytes that were never there. A mismatch refolds.
    """
    handle = segment.open("rb")
    try:
        actual = os.fstat(handle.fileno())
    except OSError:
        handle.close()
        raise
    if (actual.st_dev, actual.st_ino) != (stat.st_dev, stat.st_ino):
        handle.close()
        raise _Refold(f"{segment} changed identity between stat and open")
    return handle


def _prefix_intact(memo: _ChainMemo, chain: List[Tuple[pathlib.Path, os.stat_result, bool]]) -> bool:
    if len(memo.segments) > len(chain):
        return False
    for known, (_segment, stat, _is_live) in zip(memo.segments, chain):
        if (stat.st_dev, stat.st_ino) != (known.st_dev, known.st_ino):
            return False
        if stat.st_size < known.consumed:
            return False
        if stat.st_size == known.consumed and stat.st_mtime_ns != known.st_mtime_ns:
            return False
    return True


def _compact_row(row: Dict[str, Any], locator: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Drop an inline request body from a start row; keep a locator to re-read it."""
    request = row.get("request")
    if isinstance(request, dict) and request:
        row = {k: v for k, v in row.items() if k != "request"}
        row[REQUEST_LOCATOR_KEY] = {
            "st_dev": locator[0], "st_ino": locator[1],
            "offset": locator[2], "length": locator[3],
        }
    return row


def _fold_segment(
    memo: _ChainMemo, segment: pathlib.Path, stat: os.stat_result, is_live: bool,
    known: Optional[_Segment], *, inner: bool,
) -> _Segment:
    """Fold one segment's bytes past ``known.consumed`` into the memo's rows.

    ``inner`` marks a consumed segment that is NOT the memo's last one: rows
    from later segments were already folded after it, so a complete line
    appended to it cannot be folded in chain order and forces a refold. The
    last consumed segment may grow freely — it was the live file when its
    tail was held back, and nothing after it has been folded yet.
    """
    marker = _custody()._ROW_MARKER.encode("ascii")
    start = known.consumed if known is not None else 0
    consumed = start
    hasher = hashlib.sha256() if is_live else None
    with _open_identified(segment, stat) as handle:
        if start:
            if hasher is not None:
                # Growth alone cannot certify the consumed prefix of a growing
                # file: re-read and compare it (bounded by the rotation cap).
                hasher.update(handle.read(start))
                if not known or hasher.hexdigest() != known.prefix_sha256:
                    raise _Refold("consumed live prefix changed under the memo")
            handle.seek(start)
        for raw in handle:
            if not raw.endswith(b"\n"):
                if is_live:
                    break  # a torn live tail completes on a later call
                # An archive never completes its torn tail: consume it, count it.
                memo.torn_archive_lines += 1
                if marker in raw:
                    _remember_malformed(memo, raw)
                consumed += len(raw)
                continue
            if inner:
                raise _Refold("inner archive segment grew after it was consumed")
            offset, consumed = consumed, consumed + len(raw)
            if hasher is not None:
                hasher.update(raw)
            if marker not in raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                _remember_malformed(memo, raw)
                continue
            if not isinstance(row, dict):
                _remember_malformed(memo, raw)
                continue
            if str(row.get("type") or "").startswith(_custody()._ROW_MARKER):
                memo.rows.append(_compact_row(row, (stat.st_dev, stat.st_ino, offset, len(raw))))
        try:
            after = os.fstat(handle.fileno())
        except OSError:
            after = stat
    return _Segment(st_dev=stat.st_dev, st_ino=stat.st_ino, consumed=consumed, st_mtime_ns=after.st_mtime_ns,
                    prefix_sha256=hasher.hexdigest() if hasher is not None else "")


_MALFORMED_KEEP = 200
_MALFORMED_BYTES_KEEP = 8 * 1024 * 1024


def _remember_malformed(memo: _ChainMemo, raw: bytes) -> None:
    # WHOLE lines only: a truncated copy could drop the one id a reader needs
    # (a START_REQUESTED joined after a huge torn prefix). Past the bound the
    # record is unknown, never a shorter proof of absence (Astra 2bc1 #1).
    if (len(memo.malformed_marker_lines) >= _MALFORMED_KEEP
            or memo.malformed_bytes + len(raw) > _MALFORMED_BYTES_KEEP):
        memo.malformed_overflow = True
        return
    memo.malformed_marker_lines.append(bytes(raw))
    memo.malformed_bytes += len(raw)


def custody_rows_with_integrity(drive_root: Any, needle: str) -> Tuple[Tuple[Dict[str, Any], ...], Optional[int]]:
    """ONE refresh: the rows AND how many unparseable custody lines mention ``needle``.

    The count is None when that same refresh bypassed the memo (lenient read,
    nothing recorded) or the bounded record overflowed: no absence proof can be
    built from it. Rows and integrity come from the same read, so a check can
    never certify a different traversal than the one it judges (Astra 2bc1 #2).
    """
    key = _key(_custody().event_log_path(drive_root))
    token = str(needle or "").encode("utf-8")
    with _lock_for(key):
        memo, rows = _refresh(drive_root)
        if memo is None or memo.malformed_overflow:
            return rows, None
        return rows, sum(1 for raw in memo.malformed_marker_lines if token and token in raw)


def _advance(memo: _ChainMemo, chain: List[Tuple[pathlib.Path, os.stat_result, bool]]) -> None:
    before = len(memo.rows)
    last_known = len(memo.segments) - 1
    for index, (segment, stat, is_live) in enumerate(chain):
        known = memo.segments[index] if index < len(memo.segments) else None
        if known is not None and stat.st_size == known.consumed:
            continue
        folded = _fold_segment(memo, segment, stat, is_live, known,
                               inner=known is not None and index < last_known)
        if known is None:
            memo.segments.append(folded)
        else:
            memo.segments[index] = folded
    if len(memo.rows) != before:
        memo.generation += 1
        memo.rows_view = tuple(memo.rows)
        memo.state_cache = None


def _refresh(drive_root: Any) -> Tuple[Optional[_ChainMemo], Tuple[Dict[str, Any], ...]]:
    """Advance (or refold) the memo for ``drive_root``; ``(None, rows)`` when bypassed.

    Caller holds the per-path lock. A bypass serves today's lenient
    ``_iter_rows`` answer without caching it.
    """
    custody = _custody()
    path = custody.event_log_path(drive_root)
    key = _key(path)
    memo = _MEMOS.get(key)
    try:
        chain = _enumerate(path)
    except JsonlChainUnreadable:
        _MEMOS.pop(key, None)
        return None, tuple(custody._iter_rows(path))
    try:
        if memo is not None and not _prefix_intact(memo, chain):
            memo = None  # the consumed prefix is not the chain's prefix any more: refold
        if memo is not None:
            try:
                _advance(memo, chain)
            except _Refold:
                memo = None
        if memo is None:
            memo = _ChainMemo()
            _advance(memo, chain)
            memo.generation = 1
            memo.rows_view = tuple(memo.rows)
    except (OSError, _Refold):
        _MEMOS.pop(key, None)
        log.debug("custody memo bypassed for %s", path, exc_info=True)
        return None, tuple(custody._iter_rows(path))
    _MEMOS[key] = memo
    return memo, memo.rows_view


def custody_rows(drive_root: Any) -> Tuple[Dict[str, Any], ...]:
    """Every custody row of ``drive_root``'s chain, in chain order (read-only).

    The same dicts ``_iter_rows`` yields, except that an inline request body is
    replaced by ``request_locator``. READ-ONLY: the tuple holds the memo's own
    row objects (copying 13k rows per warm read would cost what the memo saves);
    a reader that hands rows onward copies the nested containers it exposes
    (``pending_invocations`` / ``invocation_record`` do).
    """
    key = _key(_custody().event_log_path(drive_root))
    with _lock_for(key):
        _memo, rows = _refresh(drive_root)
        return rows


def folded_state(
    drive_root: Any,
    fold: Callable[[Tuple[Dict[str, Any], ...]], Any],
    clone: Callable[[Any], Any] = copy.deepcopy,
) -> Any:
    """``fold(rows)`` over the current rows, cached per memo generation, cloned on return.

    ``clone`` copies the cached fold for the caller (the custody fold supplies
    its own container-aware copy; ``deepcopy`` is the safe default). A bypassed
    memo folds the lenient rows uncached, exactly like a replay today.
    """
    key = _key(_custody().event_log_path(drive_root))
    with _lock_for(key):
        memo, rows = _refresh(drive_root)
        if memo is None:
            return fold(rows)
        if memo.state_cache is None or memo.state_cache[0] != memo.generation:
            memo.state_cache = (memo.generation, fold(rows))
        return clone(memo.state_cache[1])


def clone_custody_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """A caller-owned copy of a folded custody state.

    ``copy.copy`` per entry plus a copy of every mutable container the fold
    writes (``resource_ref``, ``work_order_source_request``,
    ``verified_source_ranges`` and the delivery confirmations attribute), so a
    caller mutating its answer — ``lookup`` stores it in ``_CUSTODY`` and the
    ``record_*`` writers update it in place — never changes the cached fold.
    """
    clones: Dict[str, Any] = {}
    for run_id, entry in state.items():
        clone = copy.copy(entry)
        clone.resource_ref = copy.deepcopy(entry.resource_ref)
        clone.work_order_source_request = copy.deepcopy(entry.work_order_source_request)
        clone.verified_source_ranges = list(entry.verified_source_ranges)
        confirmations = getattr(entry, "_source_delivery_confirmations", None)
        if confirmations is not None:
            setattr(clone, "_source_delivery_confirmations", copy.deepcopy(confirmations))
        clones[run_id] = clone
    return clones


def read_locator_request(drive_root: Any, locator: Any, *, invocation_id: str = "") -> Optional[Dict[str, Any]]:
    """Re-read the inline ``request`` body a compacted row points at, or None.

    Bound to the file identity AND, when given, to the invocation the caller
    expects: a line that names another invocation is never returned as this
    one's body (an unknown body keeps the invocation pending; a foreign body
    would replay someone else's request).
    """
    if not isinstance(locator, dict):
        return None
    try:
        identity = (int(locator["st_dev"]), int(locator["st_ino"]))
        offset, length = int(locator["offset"]), int(locator["length"])
    except (KeyError, TypeError, ValueError):
        return None
    path = _custody().event_log_path(drive_root)
    try:
        for segment, stat, _is_live in _enumerate(path):
            if (stat.st_dev, stat.st_ino) != identity:
                continue
            with _open_identified(segment, stat) as handle:
                handle.seek(offset)
                raw = handle.read(length)
            row = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(row, dict):
                return None
            if invocation_id and str(row.get("invocation_id") or "") != invocation_id:
                return None
            body = row.get("request")
            return body if isinstance(body, dict) and body else None
    except (OSError, ValueError, JsonlChainUnreadable, _Refold):
        return None
    return None


def memo_diagnostics(drive_root: Any) -> Dict[str, Any]:
    """Facts about one memo for tests and forensics (never authority)."""
    key = _key(_custody().event_log_path(drive_root))
    with _lock_for(key):
        memo = _MEMOS.get(key)
        if memo is None:
            return {"cold": True}
        return {
            "cold": False, "generation": memo.generation, "rows": len(memo.rows),
            "segments": [(s.st_ino, s.consumed) for s in memo.segments],
            "torn_archive_lines": memo.torn_archive_lines,
        }


__all__ = [
    "REQUEST_LOCATOR_KEY", "clone_custody_state", "custody_rows", "folded_state",
    "memo_diagnostics", "read_locator_request", "reset_custody_memo",
]
