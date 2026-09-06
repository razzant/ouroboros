"""Seq-preserving compaction of the monetary usage ledger (CPL4-C6, owner 1A).

Design contract: docs/v7next/DESIGN_USAGE_COMPACTION.md. Terminal, non-review
``kind="attempt"`` chains fold into a stamped baseline block (one
``usage_baseline`` header + per-attribution ``usage_baseline_group`` rows);
the raw pre-compaction bytes move verbatim into an append-only
``archive/usage_ledger/`` segment referenced (and hash-pinned) by the header.
Nothing is deleted, in-flight rows never fold, idempotency-bearing kinds
(subscription/external/legacy) never fold, and the pass commits ONLY after
proving, on the candidate bytes, that the production aggregation renders
byte-equal results — otherwise it aborts and the ledger stays byte-identical.

Monetary exactness rule (fixed by the design note): group sums are computed as
exact ``Decimal``s of the literals stored in the file and carried on group
rows as exact-decimal JSON strings (``_number`` accepts them everywhere);
retained rows are verified decimal-identical across re-serialization, and any
non-round-trippable foreign literal aborts the pass instead of approximating.

This module sits BESIDE the substrate: it imports from ``usage_ledger`` and
``_usage_rows`` and is called INTO by ``usage_accounting.reserve_attempt``
under the held monetary lock; the substrate never imports it.
"""

from __future__ import annotations

import base64
import contextlib
import decimal
import hashlib
import json
import logging
import os
import pathlib
import stat
import threading
import time
import uuid
from decimal import Decimal, DecimalException, InvalidOperation
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from ouroboros._usage_rows import _breakdown_bucket, _summary
from ouroboros.usage_ledger import (
    ARCHIVE_SEGMENT_DIR_REL,
    LEDGER_REL,
    LOCK_REL,
    QUARANTINE_REL,
    UsageLedgerCorrupt,
    _append_bytes_fsync,
    _drive_root,
    _final_rows,
    _number,
    _read_records_locked,
    _validate_records,
    _write_bytes_atomic_fsync,
    valid_archive_rel,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# States a folded attempt chain may terminate in. In-flight (reserved/
# dispatched) finals keep their WHOLE chain in the live file.
_FOLDABLE_FINAL_STATES = frozenset({"settled", "unresolved", "released"})
_BASELINE_KINDS = frozenset({"usage_baseline", "usage_baseline_group"})
_REVIEW_KEYS = ("review_skill", "review_wave_id", "review_slot_id")
_TOKEN_SUM_FIELDS = (
    "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
)

# Thrash guard: last attempted (st_ino, st_dev, st_size) per resolved root.
# Purely per-process; the worst cost of losing it is one extra bounded pass.
_COMPACT_ATTEMPTS: Dict[str, Tuple[int, int, int]] = {}
_COMPACT_ATTEMPTS_LOCK = threading.Lock()

# Immutable-segment cache for the history readers: abs path -> (expected
# sha256, frozen attempt-id set, embedded prior header, file fingerprint, the
# moment it was verified). The fingerprint is part of the hit condition, so a
# segment deleted or rewritten after a warm read re-verifies (and fails)
# instead of answering from memory — but a fingerprint is not proof of
# identity: an in-place rewrite of the same size, inside the filesystem's
# timestamp granularity or with the mtime restored afterwards, keeps it
# whole. So a hit ALSO requires a file that has been still for a moment and
# an entry that has not been standing too long: a cached read is evidence
# with a shelf life, and past it the bytes are hashed again.
_SEGMENT_MTIME_SETTLE_SEC = 2.0
_SEGMENT_CACHE_TTL_SEC = 60.0
_SEGMENT_CACHE: Dict[
    str, Tuple[str, frozenset, Optional[Dict[str, Any]], Tuple[int, int, int, int], float]
] = {}

# The union over one WHOLE chain, keyed by the chain's identity ((archive_rel,
# sha) per hop). A reverse sweep asks the join primitive once per seal; the
# per-segment sets are cached, but re-unioning them per question is the work
# that made a bulk reconcile quadratic. Bounded: the key changes at every
# compaction and only the newest chain can ever be asked again, so a
# long-lived process must not keep one archived-id set per epoch it ever saw.
_CHAIN_UNION_CACHE_MAX = 4
_CHAIN_UNION_CACHE: Dict[Tuple[Tuple[str, str], ...], frozenset] = {}


# Money is summed in an EXPLICIT context, never the ambient one. The default
# 28-digit precision silently rounds a large-magnitude sum, and both the group
# row and the self-check that approves it are computed the same way — so a
# rounded total verifies against itself and the lost cent commits. Sixty
# digits is far past any real ledger; ``Inexact`` is trapped so that even past
# it the pass ABORTS instead of writing an approximation.
MONEY_PRECISION = 60


@contextlib.contextmanager
def _exact_money() -> Iterator[None]:
    """Decimal arithmetic that cannot silently lose a digit of money."""
    with decimal.localcontext() as context:
        context.prec = MONEY_PRECISION
        context.traps[decimal.Inexact] = True
        yield


def _decimal_of(value: Any) -> Decimal:
    """Exact decimal of a ledger monetary value (Decimal, int, or string).

    Construction is context-free by language rule, so the literal is captured
    exactly; only the arithmetic over these values needs ``_exact_money``."""
    if isinstance(value, bool):
        raise InvalidOperation
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class _Abort(Exception):
    """Internal: leave the ledger untouched (policy abort, not an I/O error)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# The typed refusal for a lock directory without kernel locks (bare NFS and
# friends): there the monetary lock is a name protocol only — exclusion the
# pass cannot prove — so it declines to rewrite the authority at all. Appends
# still land under that name protocol (disclosed best effort); the ledger
# simply stays uncompacted. The refusal is durable as well as logged: ONE
# ``usage_ledger_compaction_refused`` event per process per data root (the
# growth guard throttles the log line, not the fact), so an operator and the
# 20 MB tripwire can tell this tier apart from "nothing foldable".
NAME_TIER_REFUSAL = (
    "monetary lock is on the name tier (no kernel file locks under state/): "
    "compaction refused; appends continue under the name protocol, the ledger stays uncompacted"
)
_NAME_TIER_REFUSED: set = set()  # data roots whose refusal event this process already wrote


def _fsync_dir(path: pathlib.Path) -> None:
    """fsync a directory so entries created in it survive a power loss.

    On POSIX this is MANDATORY and its failure is fatal to the pass: an
    unsynced directory means the archive segment's name may not exist after a
    crash, and the swap that follows would then be the only surviving copy of
    a history whose raw rows just vanished. Windows has no directory handle to
    fsync (``os.open`` on a directory fails), so there it is a disclosed
    no-op — selected by the platform predicate, never by swallowing OSError.
    """
    from ouroboros.platform_layer import IS_WINDOWS

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        if IS_WINDOWS:
            return
        raise
    try:
        os.fsync(fd)
    except OSError:
        if not IS_WINDOWS:
            raise
    finally:
        os.close(fd)


def _mkdir_fsync_chain(path: pathlib.Path, root: pathlib.Path) -> None:
    """``mkdir -p`` whose WHOLE chain down from ``root`` is durable each pass.

    Syncing a directory persists the entries IT holds, so the segment's name
    lives in its own parent, that parent's name lives in ``archive/``, and
    ``archive/``'s name lives in the data root. Syncing only the levels THIS
    pass created is not enough: an earlier pass may have created a level and
    then died before its fsync, so on the retry the directories are present
    but their durability is unknown — and a crash after that retry's swap
    would lose the archive the swap depends on. Three fsyncs are cheap;
    unconditional is the only state a pass can actually prove.
    """
    path.mkdir(parents=True, exist_ok=True)
    levels = [path]
    while levels[-1] != root and levels[-1].parent != levels[-1]:
        levels.append(levels[-1].parent)
    for directory in levels:
        _fsync_dir(directory)


def _dir_fd_capable() -> bool:
    """POSIX with dir_fd/O_DIRECTORY support; Windows (or an os lacking them)
    keeps the path-based shape as a disclosed best effort, by predicate."""
    from ouroboros.platform_layer import IS_WINDOWS

    return (
        not IS_WINDOWS
        and hasattr(os, "O_DIRECTORY")
        and {os.open, os.mkdir} <= os.supports_dir_fd
    )


def _close_fds(fds: list) -> None:
    for handle in fds:
        with contextlib.suppress(OSError):
            os.close(handle)


def _archive_dir_fds(root: pathlib.Path, *, create: bool) -> list:
    """POSIX handles for [data root, ``archive/``, ``archive/usage_ledger``].

    Each level below the root is opened ``O_DIRECTORY|O_NOFOLLOW`` relative
    to the previous HANDLE (``dir_fd``), so the directory the caller then
    writes into or reads from IS the one that was checked — a link swapped in
    after any path-based look changes nothing, because no path is ever
    re-resolved. (The root itself is the anchor and may be reached through
    links legitimately.) A link at either archive level surfaces here as
    ``UsageLedgerCorrupt``.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fds: list = []
    level = root
    try:
        try:  # the anchor of the chain: an unreadable root (permissions, fd
            fds.append(os.open(str(root), os.O_RDONLY | os.O_DIRECTORY))
        except OSError as exc:  # exhaustion) must be typed too, or it escapes
            raise UsageLedgerCorrupt(f"usage archive root is not readable: {root}") from exc
        for name in ARCHIVE_SEGMENT_DIR_REL.parts:
            level = level / name
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(name, dir_fd=fds[-1])
            try:
                fds.append(os.open(name, flags, dir_fd=fds[-1]))
            except OSError as exc:
                raise UsageLedgerCorrupt(
                    f"usage archive level is not our own directory: {level}"
                ) from exc
        return fds
    except BaseException:
        _close_fds(fds)
        raise


def _write_all_fsync(fd: int, payload: bytes, path: pathlib.Path) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"short write to {path}")
        view = view[written:]
    os.fsync(fd)


def _write_new_file_fsync(path: pathlib.Path, payload: bytes, root: pathlib.Path) -> None:
    """Create-exclusive write + fsync of an archive segment AND its dir chain.

    POSIX creates the segment ``O_NOFOLLOW`` via the ``dir_fd`` handles of
    :func:`_archive_dir_fds` and fsyncs file then every directory HANDLE, so
    the chain proven durable is the one the bytes actually landed in — a link
    planted after any path-based check cannot receive monetary history.
    Windows keeps the path-based chain (no dir_fd), best effort, disclosed.
    """
    if not _dir_fd_capable():
        _mkdir_fsync_chain(path.parent, root)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(str(path), flags, 0o600)
        try:
            _write_all_fsync(fd, payload, path)
        finally:
            os.close(fd)
        _fsync_dir(path.parent)
        return
    fds = _archive_dir_fds(root, create=True)
    try:
        try:
            fd = os.open(
                path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=fds[-1],
            )
        except OSError as exc:
            raise UsageLedgerCorrupt(
                f"usage archive segment cannot be created as our own file: {path}"
            ) from exc
        try:
            _write_all_fsync(fd, payload, path)
        finally:
            os.close(fd)
        for handle in reversed(fds):  # segment dir, archive/, data root
            os.fsync(handle)
    finally:
        _close_fds(fds)


def _swap_ledger_fsync(
    path: pathlib.Path, payload: bytes, raw: bytes, beat: Callable[[], None]
) -> None:
    """Atomically replace the live ledger with the verified candidate bytes.

    The proofs run immediately BEFORE the rename — after the temp bytes are
    durable, at the last instant the replace can still be refused, because
    writing and fsyncing the candidate temp can take arbitrarily long and any
    proof taken before the swap began is stale by the rename. Ownership FIRST
    (``beat`` aborts the pass if the lock stopped being ours while the temp
    was written — a snapshot answered while robbed is a meaningless answer),
    THEN the live file must still be the snapshot ``raw`` this candidate
    folded: the recheck-to-replace gap is where an append under a broken lock
    would be silently erased, so the last look happens inside the swap itself
    — and ownership is proven AGAIN after that look, so the only interval
    left between the last proof and the rename is the syscall, not the
    milliseconds of a full-file compare. AFTER the rename the pass re-reads
    what landed: the receipt describes bytes that are actually AT the path,
    so a rename that reported success without landing (or was immediately
    written over) is never logged as a compaction. The one interval no proof
    covers is the rename syscall itself: a charge an out-of-protocol holder
    lands on the OLD inode inside it is erased by that rename, and nothing at
    the path shows it afterwards. On POSIX the old inode is held open across
    the swap (Windows cannot hold the destination open through ``os.replace``:
    there the loss stays silent, disclosed), so the bytes that landed beyond
    the snapshot are still readable AFTER the fact: they are preserved in the
    quarantine file — which flags ``integrity_degraded`` — and the pass raises
    typed instead of returning a receipt over an erased charge. Detected by
    size, never re-appended; a same-size in-place rewrite inside that one
    syscall is not a landed charge and is not seen.
    """
    from ouroboros.platform_layer import IS_WINDOWS

    def witness_is_the_path() -> bool:
        try:  # the witness fd must be the very inode the proof just licensed
            return old_fd is None or os.fstat(old_fd)[1:3] == os.stat(path)[1:3]
        except OSError:
            return False

    def owned_and_intact() -> bool:
        beat()  # raises _Abort on a lost hold; the temp is cleaned up en route
        intact = _snapshot_intact(path, raw) and witness_is_the_path()
        beat()  # and once more AFTER the look: the rename is the next syscall
        return intact

    try:
        old_fd = None if IS_WINDOWS else os.open(str(path), os.O_RDONLY)  # the only witness left after the rename
    except OSError as exc:  # the ledger vanished under the held lock: an abort by policy, not a bare OSError
        raise _Abort(f"ledger not openable before the swap: {exc}") from exc
    try:
        if not _write_bytes_atomic_fsync(path, payload, precondition=owned_and_intact):
            raise _Abort("ledger changed between the re-check and the replace")
        _fsync_dir(path.parent)
        erased = b"" if old_fd is None else os.pread(old_fd, max(0, os.fstat(old_fd).st_size - len(raw)), len(raw))
    finally:
        if old_fd is not None:
            os.close(old_fd)
    if erased:
        _append_bytes_fsync(path.with_name(QUARANTINE_REL.name), (json.dumps({
            "ts": utc_now_iso(), "source": str(path), "raw_base64": base64.b64encode(erased).decode("ascii"),
            "reason": "erased by the compaction swap: landed between its last ownership proof and the rename",
        }, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        raise UsageLedgerCorrupt(
            f"usage ledger swap erased {len(erased)} bytes appended between the last proof and the rename "
            f"(quarantined, integrity degraded): {path}"
        )
    if path.read_bytes() != payload:
        raise UsageLedgerCorrupt(f"usage ledger swap did not land the approved bytes: {path}")


def _beat(heartbeat: Callable[[], bool]) -> None:
    """Renew the held monetary lock at a checkpoint; ABORT if it is not ours.

    The heartbeat answers ownership, and a pass that keeps working after the
    lock left it is exactly the pass that swaps its snapshot over a charge the
    new owner appended. So a ``False`` — or an answer we cannot get at all —
    abandons the pass. That is never a failed reservation: the caller reads an
    abort as "not compacted this time" and the ledger stays byte-identical.
    There is no "no heartbeat" case: an absent one is a caller defect, and it
    aborts here (a TypeError from the call) rather than proving nothing.
    """
    try:
        owned = heartbeat()
    except Exception as exc:
        raise _Abort(f"monetary lock heartbeat failed: {type(exc).__name__}") from exc
    if not owned:
        raise _Abort("monetary lock ownership lost mid-pass")


def _snapshot_intact(ledger_path: pathlib.Path, raw: bytes) -> bool:
    """Whether the live ledger is still EXACTLY the snapshot being compacted.

    The swap replaces the WHOLE file, so any row appended after the snapshot
    would be silently dropped — a settled charge erased, a budget under-count,
    a replayable double charge. The lock makes that impossible in the normal
    case; this makes it impossible in the abnormal one (a lock broken by age,
    a foreign writer, a manual repair). Re-read under the same held lock and
    refuse the swap on ANY difference: the price of a lost race is a skipped
    compaction pass, never a lost row.
    """
    try:
        if os.stat(ledger_path).st_size != len(raw):
            return False
        return ledger_path.read_bytes() == raw
    except OSError:
        return False


def _dumps_row(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_weight(row: Dict[str, Any]) -> int:
    if str(row.get("kind") or "") == "usage_baseline_group":
        return max(1, int(row.get("folded_attempt_count") or 1))
    return 1


def _group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """The attribution tuple that keeps every aggregation branch homogeneous."""
    pricing_known = row.get("pricing_known")
    return (
        str(row.get("state") or ""),
        str(row.get("model") or ""),
        str(row.get("provider") or ""),
        str(row.get("category") or ""),
        str(row.get("source") or ""),
        str(row.get("task_id") or ""),
        str(row.get("root_task_id") or ""),
        str(row.get("parent_task_id") or ""),
        str(row.get("prompt_cache_ttl") or ""),
        row.get("cost_usd") is not None,
        bool(row.get("cost_final")),
        pricing_known if isinstance(pricing_known, bool) else None,
        row.get("reservation_upper_bound_usd") is not None,
    )


class _Group:
    __slots__ = ("count", "cost", "bound", "tokens", "root_limit")

    def __init__(self) -> None:
        self.count = 0
        self.cost: Optional[Decimal] = None
        self.bound: Optional[Decimal] = None
        self.tokens: Dict[str, Optional[int]] = {field: None for field in _TOKEN_SUM_FIELDS}
        self.root_limit: Optional[Decimal] = None

    def absorb(self, row: Dict[str, Any]) -> None:
        """Fold one FINAL decimal-parsed row (attempt final or prior group)."""
        self.count += _row_weight(row)
        cost = row.get("cost_usd")
        if cost is not None:
            self.cost = (self.cost or Decimal(0)) + _decimal_of(cost)
        bound = row.get("reservation_upper_bound_usd")
        if bound is not None:
            self.bound = (self.bound or Decimal(0)) + _decimal_of(bound)
        for field in _TOKEN_SUM_FIELDS:
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise _Abort(f"non-integer token field {field}")
            self.tokens[field] = (self.tokens[field] or 0) + max(0, value)
        limit = row.get("root_limit_usd")
        if limit is not None and _number(limit) is not None:
            limit_dec = _decimal_of(limit)
            self.root_limit = (
                limit_dec if self.root_limit is None else min(self.root_limit, limit_dec)
            )


def _render_fingerprint(finals: list) -> Dict[str, Any]:
    """The production-aggregation surfaces budget/display actually consume.

    Mirrors the composition of ``usage_projection`` (global summary, per-root
    summaries + min known ``root_limit_usd``) and ``usage_breakdown`` (global
    bucket, per-axis buckets with the legacy/empty-key unattributed rule),
    built from the SAME ``_summary``/``_breakdown_bucket`` production
    functions. Compared before/after on the candidate bytes; any inequality
    aborts the compaction.
    """
    per_root: Dict[str, Any] = {}
    grouped_roots: Dict[str, list] = {}
    for row in finals:
        rid = str(row.get("root_task_id") or "")
        if rid:
            grouped_roots.setdefault(rid, []).append(row)
    for rid in sorted(grouped_roots):
        rows = grouped_roots[rid]
        known = [
            value
            for value in (_number(row.get("root_limit_usd")) for row in rows)
            if value is not None
        ]
        per_root[rid] = (_summary(rows), min(known) if known else None)

    def grouped(field: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        groups: Dict[str, list] = {}
        unattributed: list = []
        for row in finals:
            key = str(row.get(field) or "")
            if str(row.get("kind") or "") in {"legacy_metadata", "legacy_delta"} or not key:
                unattributed.append(row)
            else:
                groups.setdefault(key, []).append(row)
        return (
            {key: _breakdown_bucket(groups[key]) for key in sorted(groups)},
            _breakdown_bucket(unattributed),
        )

    return {
        "summary": _summary(finals),
        "by_root": per_root,
        "breakdown": _breakdown_bucket(finals),
        "axes": {
            field: grouped(field)
            for field in ("model", "provider", "category", "task_id", "root_task_id")
        },
    }


def _parse_ledger_lines(raw: bytes) -> Tuple[list, list]:
    """(float_rows, decimal_rows) for the non-empty lines of ``raw``."""
    float_rows: list = []
    decimal_rows: list = []
    for chunk in raw.splitlines():
        line = chunk.strip(b"\r").strip()
        if not line:
            continue
        text = line.decode("utf-8")
        float_rows.append(json.loads(text))
        decimal_rows.append(json.loads(text, parse_float=Decimal))
    return float_rows, decimal_rows


def _foldable_attempt_ids(records: list) -> set:
    """Attempt ids whose whole chain folds: terminal, plain ``attempt`` kind,
    no review attribution — plus prior baseline group/header rows (re-folded)."""
    finals = _final_rows(records)
    foldable: set = set()
    for attempt_id, row in finals.items():
        kind = str(row.get("kind") or "attempt")
        if kind in _BASELINE_KINDS:
            foldable.add(attempt_id)
            continue
        if kind != "attempt":
            continue
        if str(row.get("state") or "") not in _FOLDABLE_FINAL_STATES:
            continue
        if any(str(row.get(key) or "") for key in _REVIEW_KEYS):
            continue
        if isinstance(row.get("cost_usd"), bool) or isinstance(
            row.get("reservation_upper_bound_usd"), bool
        ):
            continue  # fail-safe: malformed monetary value never folds
        foldable.add(attempt_id)
    return foldable


def _build_candidate(
    records: list, decimal_records: list, raw: bytes, beat: Callable[[], None],
) -> Tuple[bytes, Dict[str, Any]]:
    """Fold ``records`` into the candidate bytes + commit receipt.

    ``beat`` renews the held lock at the checkpoints of the two row walks: on
    a multi-megabyte ledger this loop, not the commit, is where the pass can
    outlive its own staleness window — so it is required, like the entry
    points' heartbeat, never a default that silently degrades to a no-op.
    """
    foldable = _foldable_attempt_ids(records)
    decimal_finals = _final_rows(decimal_records)
    prior_header = next(
        (row for row in records if str(row.get("kind") or "") == "usage_baseline"), None
    )

    groups: Dict[Tuple[Any, ...], _Group] = {}
    folded_row_count = 0
    folded_attempt_count = 0
    for index, attempt_id in enumerate(foldable):
        if index % 4096 == 0:
            beat()
        final = decimal_finals[attempt_id]
        if str(final.get("kind") or "") == "usage_baseline":
            continue  # the prior stamp is superseded, not aggregated
        groups.setdefault(_group_key(final), _Group()).absorb(final)
        folded_attempt_count += _row_weight(final)
    for row in records:
        if str(row.get("attempt_id") or "") in foldable:
            folded_row_count += 1
    if folded_row_count == 0 or not groups:
        raise _Abort("nothing foldable")

    baseline_id = f"baseline-{uuid.uuid4().hex[:12]}"
    epoch = 1
    if prior_header is not None:
        epoch = max(1, int(prior_header.get("compaction_epoch") or 0)) + 1
    now = utc_now_iso()
    stamp = now.replace("-", "").replace(":", "").replace("+00:00", "Z")
    archive_rel = str(
        ARCHIVE_SEGMENT_DIR_REL / f"segment_ep{epoch:04d}_{stamp}_{uuid.uuid4().hex[:8]}.jsonl"
    ).replace(os.sep, "/")
    source_sha256 = hashlib.sha256(raw).hexdigest()

    group_rows: list = []
    for index, key in enumerate(sorted(groups, key=repr), start=1):
        group = groups[key]
        (state, model, provider, category, source, task_id, root_task_id,
         parent_task_id, ttl, cost_known, cost_final, pricing_known,
         bound_known) = key
        row: Dict[str, Any] = {
            "kind": "usage_baseline_group",
            "attempt_id": f"{baseline_id}-g{index:04d}",
            "state": state,
            "model": model,
            "provider": provider,
            "category": category,
            "source": source,
            "task_id": task_id,
            "root_task_id": root_task_id,
            "parent_task_id": parent_task_id,
            "review_skill": "",
            "review_wave_id": "",
            "review_slot_id": "",
            "baseline_id": baseline_id,
            "folded_attempt_count": group.count,
            "cost_final": cost_final,
            "ts": now,
        }
        if ttl:
            row["prompt_cache_ttl"] = ttl
        if pricing_known is not None:
            row["pricing_known"] = pricing_known
        if cost_known:
            row["cost_usd"] = format(group.cost or Decimal(0), "f")
        if bound_known:
            row["reservation_upper_bound_usd"] = format(group.bound or Decimal(0), "f")
        for field in _TOKEN_SUM_FIELDS:
            if group.tokens[field] is not None:
                row[field] = group.tokens[field]
        if group.root_limit is not None:
            row["root_limit_usd"] = format(group.root_limit, "f")
        group_rows.append(row)

    retained_lines: list = []
    retained_count = 0
    next_seq = 1 + len(group_rows)
    for index, (float_row, decimal_row) in enumerate(zip(records, decimal_records)):
        if index % 4096 == 0:
            beat()
        attempt_id = str(float_row.get("attempt_id") or "")
        kind = str(float_row.get("kind") or "attempt")
        if attempt_id in foldable or kind == "usage_baseline":
            continue
        next_seq += 1
        retained_count += 1
        updated = dict(float_row)
        updated["pre_compaction_seq"] = int(float_row.get("seq") or 0)
        updated["seq"] = next_seq
        line = _dumps_row(updated)
        expected = dict(decimal_row)
        expected["pre_compaction_seq"] = updated["pre_compaction_seq"]
        expected["seq"] = next_seq
        if json.loads(line, parse_float=Decimal) != expected:
            # A foreign literal that does not round-trip through a double:
            # never approximate the monetary history — keep the file as is.
            raise _Abort(f"non-round-trippable literal in retained row seq={float_row.get('seq')}")
        retained_lines.append(line)

    header = {
        "kind": "usage_baseline",
        "attempt_id": baseline_id,
        "state": "settled",
        "seq": 1,
        "ts": now,
        "baseline_id": baseline_id,
        "compaction_epoch": epoch,
        "archive_rel": archive_rel,
        "source_sha256": source_sha256,
        "source_size_bytes": len(raw),
        "source_row_count": len(records),
        "source_first_seq": int(records[0].get("seq") or 1),
        "source_last_seq": int(records[-1].get("seq") or len(records)),
        "folded_row_count": folded_row_count,
        "folded_attempt_count": folded_attempt_count,
        "group_count": len(group_rows),
        "retained_row_count": retained_count,
    }
    for offset, row in enumerate(group_rows, start=2):
        row["seq"] = offset
    lines = [_dumps_row(header), *(_dumps_row(row) for row in group_rows), *retained_lines]
    candidate = ("\n".join(lines) + "\n").encode("utf-8")

    receipt = {
        "baseline_id": baseline_id,
        "compaction_epoch": epoch,
        "archive_rel": archive_rel,
        "source_sha256": source_sha256,
        "source_size_bytes": len(raw),
        "compacted_size_bytes": len(candidate),
        "source_row_count": len(records),
        "folded_row_count": folded_row_count,
        "folded_attempt_count": folded_attempt_count,
        "group_count": len(group_rows),
        "retained_row_count": retained_count,
    }
    return candidate, receipt


def compact_usage_ledger_locked(
    root: pathlib.Path | str,
    *,
    heartbeat: Callable[[], bool],
) -> Optional[Dict[str, Any]]:
    """One compaction pass. MUST be called under the held monetary ledger lock.

    ``heartbeat`` is the lock renewal yielded by ``usage_ledger._locked``: a
    pass over a multi-megabyte ledger can outlive the lock's staleness window,
    and a lock stolen mid-pass is the one way a swap could drop a concurrently
    appended charge. It is REQUIRED, not defaulted: every ownership proof in
    this module runs through it, so an omitted one would silently turn each of
    them into a no-op and swap the monetary authority unproven. A caller that
    drops it gets a TypeError, not a pass that runs blind.

    Returns the commit receipt, or ``None`` when the pass aborts by policy
    (nothing foldable, no byte gain, any verification inequality, a live
    ledger that changed under us, or a lock that stopped being ours) — an
    abort leaves the ledger byte-identical. I/O errors during the commit steps
    propagate; the archive segment is durable BEFORE the live file is touched,
    so a crash at any point leaves a valid ledger.
    """
    from ouroboros.platform_layer import kernel_file_locks_enforced

    root = pathlib.Path(_drive_root(root))
    if not kernel_file_locks_enforced(root / LOCK_REL):
        log.warning("usage-ledger compaction refused: %s", NAME_TIER_REFUSAL)
        told = str(root.resolve(strict=False))  # one data root, however it is spelled
        if told not in _NAME_TIER_REFUSED:
            try:  # only a row that LANDED is "already told": append_jsonl reports its
                if append_jsonl(root / "logs" / "events.jsonl", {  # exhausted retries as
                    "type": "usage_ledger_compaction_refused", "ts": utc_now_iso(),  # False (and
                    "reason": "name_tier", "lock_dir": str((root / LOCK_REL).parent),  # logs it)
                }):
                    _NAME_TIER_REFUSED.add(told)
            except Exception:
                log.exception("Failed to emit usage_ledger_compaction_refused event")
        return None
    ledger_path = root / LEDGER_REL
    records = _read_records_locked(root)  # owns quarantine of a torn tail
    if not records:
        return None
    try:
        raw = ledger_path.read_bytes()
    except OSError:
        return None
    def beat() -> None:
        _beat(heartbeat)

    try:
        with _exact_money():
            beat()
            float_rows, decimal_rows = _parse_ledger_lines(raw)
            if len(float_rows) != len(records):
                raise _Abort("post-read line drift")
            candidate, receipt = _build_candidate(float_rows, decimal_rows, raw, beat)
            if len(candidate) >= len(raw):
                raise _Abort("no byte gain")
            beat()
            candidate_records, candidate_decimals = _parse_ledger_lines(candidate)
            _validate_records(candidate_records)
            beat()
            finals_before = list(_final_rows(float_rows).values())
            finals_after = list(_final_rows(candidate_records).values())
            if _render_fingerprint(finals_before) != _render_fingerprint(finals_after):
                raise _Abort("aggregation fingerprint mismatch")
            beat()

            def decimal_totals(rows: list) -> Tuple[Decimal, Decimal]:
                cost = Decimal(0)
                bound = Decimal(0)
                for row in _final_rows(rows).values():
                    if str(row.get("kind") or "") == "usage_baseline":
                        continue
                    value = row.get("cost_usd")
                    if value is not None and str(row.get("state") or "") == "settled":
                        cost += _decimal_of(value)
                    upper = row.get("reservation_upper_bound_usd")
                    if upper is not None:
                        bound += _decimal_of(upper)
                return cost, bound

            if decimal_totals(decimal_rows) != decimal_totals(candidate_decimals):
                raise _Abort("decimal money totals mismatch")
        _beat(heartbeat)
    except _Abort as abort:
        log.info("usage-ledger compaction skipped: %s", abort.reason)
        return None
    except (UsageLedgerCorrupt, DecimalException, ValueError, TypeError, KeyError) as exc:
        # Never let a compaction defect become a monetary failure: abort clean.
        log.warning("usage-ledger compaction aborted: %s: %s", type(exc).__name__, exc)
        return None

    # Commit: archive first (durable before the live file is touched). The
    # literal path chain here is the scanner-visible writer of the
    # archive/usage_ledger plane (docs/PERSISTENCE.md row). Ownership is
    # re-proven IMMEDIATELY before each snapshot look — including the final
    # look inside the swap itself, after the candidate temp is durable and
    # right before the rename: a re-check answered while the lock already
    # belongs to someone else is a meaningless answer, and the rename is the
    # irreversible step it licenses.
    try:
        try:
            _archive_dir_bounded(root)  # never write history through a link
        except UsageLedgerCorrupt as exc:
            raise _Abort(str(exc)) from exc
        beat()
        if not _snapshot_intact(ledger_path, raw):
            log.warning("usage-ledger compaction abandoned before archive: ledger changed under the lock")
            return None
        segment_name = pathlib.PurePosixPath(receipt["archive_rel"]).name
        try:
            _write_new_file_fsync(root / "archive" / "usage_ledger" / segment_name, raw, root)
        except UsageLedgerCorrupt as exc:
            raise _Abort(str(exc)) from exc  # e.g. a link planted mid-pass
        beat()
        if not _snapshot_intact(ledger_path, raw):
            # A row landed between the snapshot and here. Swapping now would
            # erase it; the written segment is an orphan (never referenced).
            log.warning("usage-ledger compaction abandoned before swap: ledger changed under the lock")
            return None
        _swap_ledger_fsync(ledger_path, candidate, raw, beat)
    except _Abort as abort:
        log.info("usage-ledger compaction skipped: %s", abort.reason)
        return None
    try:
        append_jsonl(
            root / "logs" / "events.jsonl",
            {"type": "usage_ledger_compacted", "ts": utc_now_iso(), **receipt},
        )
    except Exception:
        log.exception("Failed to emit usage_ledger_compacted event")
    log.info(
        "usage ledger compacted: %s -> %s bytes, folded %s rows (%s attempts) into %s groups, archive %s",
        receipt["source_size_bytes"], receipt["compacted_size_bytes"],
        receipt["folded_row_count"], receipt["folded_attempt_count"],
        receipt["group_count"], receipt["archive_rel"],
    )
    return receipt


def maybe_compact_usage_ledger_locked(
    root: pathlib.Path | str,
    *,
    heartbeat: Callable[[], bool],
) -> bool:
    """Opportunistic trigger on the monetary write path (under the held lock).

    ``os.stat`` fast-path below ``config.USAGE_LEDGER_COMPACT_BYTES``; a
    per-process growth guard throttles re-attempts after an unprofitable or
    aborted pass. Every failure is contained: this never raises into the
    caller's reservation (a corrupt ledger still fails in the normal read)."""
    try:
        root = pathlib.Path(_drive_root(root))
    except Exception:
        return False
    ledger_path = root / LEDGER_REL
    try:
        stat = os.stat(ledger_path)
    except OSError:
        return False
    from ouroboros import config

    if stat.st_size < int(config.USAGE_LEDGER_COMPACT_BYTES):
        return False
    key = str(root.resolve(strict=False))
    with _COMPACT_ATTEMPTS_LOCK:
        prior = _COMPACT_ATTEMPTS.get(key)
    if prior is not None and prior[:2] == (stat.st_ino, stat.st_dev) and (
        stat.st_size < prior[2] + int(config.USAGE_LEDGER_COMPACT_RETRY_GROWTH_BYTES)
    ):
        return False
    receipt: Optional[Dict[str, Any]] = None
    try:
        receipt = compact_usage_ledger_locked(root, heartbeat=heartbeat)
    except Exception:
        log.exception("usage-ledger compaction pass raised; the reservation continues on the ledger as it stands")
    if receipt is not None:
        with _COMPACT_ATTEMPTS_LOCK:
            _COMPACT_ATTEMPTS.pop(key, None)
        return True
    with _COMPACT_ATTEMPTS_LOCK:
        _COMPACT_ATTEMPTS[key] = (stat.st_ino, stat.st_dev, stat.st_size)
    return False


# --- History readers (CPL-5 reverse-sweep join surface; audits) --------------


def _live_baseline_header(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The live ledger's leading baseline header, if the file is compacted.

    Lock-free by design: appends never touch line 1 and the compactor swaps
    the file atomically, so the first line is always a complete row of either
    generation. ``None`` means a readable leading row that is not a baseline
    stamp: either no compaction has happened, or one has and its stamp is gone
    (a restore from a backup older than the last pass). This function cannot
    tell those apart and does not try — the archive does, in the epoch anchor,
    which runs on a stamp-less file too. A row that cannot be read AT ALL is
    corruption and says so: reporting it as "not compacted" would hand the
    CPL-5 sweep an empty archive and let it call a folded attempt an orphan
    seal.
    """
    try:
        with open(root / LEDGER_REL, "rb") as handle:
            first = handle.readline()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UsageLedgerCorrupt(f"usage ledger unreadable: {root / LEDGER_REL}") from exc
    line = first.strip(b"\r\n").strip()
    if not line:
        return None
    try:
        row = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UsageLedgerCorrupt("unreadable leading usage ledger row") from exc
    if not isinstance(row, dict):
        raise UsageLedgerCorrupt("leading usage ledger row is not an object")
    return row if str(row.get("kind") or "") == "usage_baseline" else None


def _archive_dir_bounded(root: pathlib.Path) -> pathlib.Path:
    """The archive directory, proven to BE inside this data root.

    ``archive/`` and ``archive/usage_ledger`` are ours. A symlink at either
    level turns "the segment resolves next to the archive directory" into a
    tautology — both sides resolve through the same link — so the monetary
    history could be read from, or written into, anywhere on the host.
    Neither level may be a link, and the resolved directory must be exactly
    the archive path of the resolved data root.
    """
    directory = root / ARCHIVE_SEGMENT_DIR_REL
    try:
        for level in (directory.parent, directory):
            if level.is_symlink():
                raise UsageLedgerCorrupt(f"usage archive path is a symlink: {level}")
    except OSError as exc:  # pathlib re-raises all but ENOENT/ENOTDIR/EBADF/ELOOP: type it
        raise UsageLedgerCorrupt(f"usage archive path cannot be inspected: {directory}") from exc
    resolved = pathlib.Path(os.path.realpath(directory))
    if resolved != pathlib.Path(os.path.realpath(root)) / ARCHIVE_SEGMENT_DIR_REL:
        raise UsageLedgerCorrupt(f"usage archive directory escapes its data root: {directory}")
    return resolved


def _segment_path(root: pathlib.Path, archive_rel: str) -> pathlib.Path:
    """Resolve a header's ``archive_rel`` INSIDE the archive directory or fail.

    Three independent bounds: the textual shape the substrate declares legal,
    an archive directory that is genuinely this data root's own, and the
    RESOLVED location of the segment (so a symlink planted in the archive
    cannot make a file elsewhere on the host count as archived history)."""
    if not valid_archive_rel(archive_rel):
        raise UsageLedgerCorrupt(
            f"usage baseline archive reference is not bounded: {archive_rel!r}"
        )
    archive_dir = _archive_dir_bounded(root)
    try:  # realpath, not Path.resolve: the non-strict realpath never raises on a symlink loop
        path = pathlib.Path(os.path.realpath(root / archive_rel))
        linked = (root / archive_rel).is_symlink()
    except (OSError, RuntimeError) as exc:  # a refused lstat, a loop: UNKNOWN, typed — never bare
        raise UsageLedgerCorrupt(f"usage archive segment cannot be inspected: {archive_rel!r}") from exc
    if path.parent != archive_dir or linked:
        raise UsageLedgerCorrupt(
            f"usage archive segment escapes the archive directory: {archive_rel!r}"
        )
    return path


def _open_archive_entry(path: pathlib.Path, dir_fd: Optional[int]) -> int:
    """Open one archive entry for reading: POSIX opens ``path.name``
    ``O_NOFOLLOW`` relative to the held archive handle, so the file read is a
    plain file inside the directory that was checked — a link planted after
    any path-based look is an open error, never a read through it — and
    ``O_NONBLOCK``, so a FIFO planted there (no writer: a blocking open never
    returns) opens at once and is classified by the caller's ``fstat``. Without
    a handle (Windows) the open is path-based, best effort, by the predicate —
    and carries the same non-blocking flag where the platform has one."""
    if dir_fd is None:
        return os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)


def _load_segment(
    root: pathlib.Path, header: Dict[str, Any], dir_fd: Optional[int]
) -> Tuple[frozenset, Optional[Dict[str, Any]]]:
    """Read one archived segment named (and fully described) by ``header``.

    Segments are immutable, so a verified read is cached — but the cache is
    keyed on what the file IS, not merely on what was once read from that
    path, and the hit expires: a deleted or rewritten segment must surface as
    corruption even in a process that already loaded it, or an audit keeps
    answering "logged" from history that is no longer there.
    """
    expected_sha256 = str(header.get("source_sha256") or "")
    path = _segment_path(root, str(header.get("archive_rel") or ""))
    key = str(path)
    try:
        fd = _open_archive_entry(path, dir_fd)
    except OSError as exc:
        raise UsageLedgerCorrupt(f"usage archive segment unreadable: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):  # a directory/FIFO/device standing where a segment is named
            raise UsageLedgerCorrupt(f"usage archive segment is not a regular file: {path}")
        fingerprint = (info.st_ino, info.st_dev, info.st_size, info.st_mtime_ns)
        now = time.time()
        cached = _SEGMENT_CACHE.get(key)
        if (
            cached is not None
            and cached[0] == expected_sha256
            and cached[3] == fingerprint
            and (now - info.st_mtime) > _SEGMENT_MTIME_SETTLE_SEC
            and (now - cached[4]) <= _SEGMENT_CACHE_TTL_SEC
        ):
            return cached[1], cached[2]
        chunks: list = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    except OSError as exc:  # the CPL-5 sweep maps typed corruption to UNKNOWN; a bare OSError escapes it
        raise UsageLedgerCorrupt(f"usage archive segment unreadable: {path}") from exc
    finally:
        os.close(fd)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise UsageLedgerCorrupt(f"usage archive segment hash mismatch: {path}")
    if len(payload) != int(header.get("source_size_bytes") or -1):
        raise UsageLedgerCorrupt(f"usage archive segment size disagrees with its header: {path}")
    try:
        rows, _ = _parse_ledger_lines(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise UsageLedgerCorrupt(f"corrupt usage archive segment row: {path}") from exc
    if len(rows) != int(header.get("source_row_count") or -1):
        raise UsageLedgerCorrupt(f"usage archive segment row count disagrees with its header: {path}")
    # A segment IS a former generation of the ledger: hold it to the same
    # structural authority (dense seq, legal transitions, well-formed rows)
    # instead of scraping any JSON object that carries an ``attempt_id``.
    _validate_records(rows)
    ids = {str(row.get("attempt_id") or "") for row in rows}
    ids.discard("")
    prior_header = rows[0] if rows and str(rows[0].get("kind") or "") == "usage_baseline" else None
    frozen = frozenset(ids)
    _SEGMENT_CACHE[key] = (expected_sha256, frozen, prior_header, fingerprint, now)
    return frozen, prior_header


def _no_newer_archived_epoch(
    root: pathlib.Path, live_header: Optional[Dict[str, Any]], walked: set, dir_fd: Optional[int]
) -> None:
    """Refuse a live file the archive itself knows to be an older generation.

    The chain walk proves every hop, but it starts wherever the live header
    points: re-pointing that header at an older GENUINE segment while also
    lowering the mutable ``compaction_epoch`` yields a chain that is valid and
    simply short. The generations the forgery orphaned are still on disk, so
    the archive anchors the live epoch. A segment produced by epoch N embeds
    the epoch N-1 header (none at all for epoch 1), so a segment's generation
    is read from its CONTENT, never from its name.

    The scan runs in the directory the chain was walked in — through the held
    ``dir_fd`` on POSIX, entries opened relative to it, so a directory swapped
    after the walk cannot hide a generation from it; by path elsewhere. An
    entry the scan cannot list, open or read is typed corruption: the scan did
    not complete, so the history question is UNKNOWN. An entry that opens but
    is not a regular file (a stray directory, a FIFO) is no segment — segments
    are regular files by construction, no generation lives there — and is
    skipped, not corruption; without a held handle that classification happens
    BEFORE the open, which is the step a directory refuses (Windows) and a
    writer-less FIFO blocks on. A first row that reads but does not parse is a
    torn segment from a crashed write: no evidence of any generation, left to
    the walk, which verifies every segment the answer actually depends on.

    A newer generation has exactly ONE legal shape: the uncommitted orphan of
    the live generation. A pass writes its segment BEFORE the swap, so that
    segment is the byte-for-byte copy of the live file at that instant, and
    the live file only grows behind it — the orphan's bytes are still a PREFIX
    of it, and every id it holds is live. That prefix is the test, read from
    the descriptor the entry was classified through (one open per entry: a
    name re-opened in between could name a different file). Matching
    only the segment's leading row against the live header was not: a previous
    generation RESTORED over the live file (a backup, a rescue snapshot) has
    the same leading row and also carries every attempt the rolled-back
    compaction folded — ids that exist nowhere else — so the exemption hid
    them from the join, the one verdict this surface may never reach. A live
    file with NO stamp is the same question with the floor at zero: its only
    legal companion is a pre-compaction generation (leading row an ordinary
    attempt row, so produced by epoch 1) that is likewise still its prefix.
    """
    live_epoch = int(live_header.get("compaction_epoch") or 0) if live_header else 0
    directory = root / ARCHIVE_SEGMENT_DIR_REL
    try:
        for name in sorted(os.listdir(directory if dir_fd is None else dir_fd)):
            if name in walked:
                continue
            if dir_fd is None and not stat.S_ISREG(os.stat(directory / name).st_mode):
                continue  # no held handle: classify BEFORE the open, which is what a
            fd = _open_archive_entry(directory / name, dir_fd)  # directory refuses there
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    continue
                try:
                    row = json.loads(os.read(fd, 1 << 16).split(b"\n", 1)[0].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                prior = row.get("compaction_epoch") if str(row.get("kind") or "") == "usage_baseline" else 0
                if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
                    continue
                if prior + 1 <= live_epoch:
                    continue
                os.lseek(fd, 0, os.SEEK_SET)  # compared from the descriptor just classified, never a re-open
                with open(root / LEDGER_REL, "rb") as live:  # an orphan of the live generation, or a rolled-back one
                    while chunk := os.read(fd, 1 << 20):
                        if live.read(len(chunk)) != chunk:
                            raise UsageLedgerCorrupt(
                                f"usage archive holds a generation newer than the live baseline: {name}"
                            )
            finally:
                os.close(fd)
    except OSError as exc:
        raise UsageLedgerCorrupt(f"usage archive anchor scan could not complete: {exc}") from exc


def _union_segment_ids(segment_ids: list) -> frozenset:
    ids: set = set()
    for chunk in segment_ids:
        ids |= chunk
    return frozenset(ids)


def archived_attempt_ids(root: pathlib.Path | str | None = None) -> frozenset:
    """Every ``attempt_id`` recorded in the archived ledger segments.

    Walks the tamper-evident chain: the live header names (and hash-pins) the
    newest segment; each segment's own leading header names the one before it.
    Because the source of epoch N is exactly the file epoch N-1 produced, the
    chain's epochs must step down by one and end at epoch 1 with a segment
    that embeds no header — so re-pointing a live header at an older genuine
    segment (dropping the epochs between) is corruption, not a shorter
    history. Because that stamp's own epoch is mutable too, the archive
    anchors it: no segment on disk may carry a generation newer than the live
    file's own, except the uncommitted orphan of that generation — the pre-swap
    copy of the live file, still a prefix of it. The anchor runs whether or not
    the live file carries a stamp, so a stamp that was REMOVED (a restore from
    a backup taken before the last compaction) is a corruption verdict instead
    of an archive nobody consulted.
    Segments are immutable, so per-segment reads and the union over a given
    chain are cached. An unreadable (or not-a-regular-file), hash-mismatched,
    mis-stepped, cyclic or out-anchored chain raises ``UsageLedgerCorrupt`` —
    the CPL-5 reverse sweep must treat that as its existing UNKNOWN /
    skip-pass state, never as evidence of an orphan."""
    root = pathlib.Path(_drive_root(root))
    live_header = _live_baseline_header(root)
    if live_header is None:  # no stamp: only the kernel's exact "no archive directory" ends
        for level in ((root / ARCHIVE_SEGMENT_DIR_REL).parent, root / ARCHIVE_SEGMENT_DIR_REL):
            try:  # a link at either level — dangling included — is the stamped reader's refusal, not ENOENT
                if stat.S_ISLNK(os.lstat(level).st_mode):
                    raise UsageLedgerCorrupt(f"usage archive path is a symlink: {level}")
            except FileNotFoundError:
                continue  # absent: the exact-ENOENT question below decides
            except OSError as exc:
                raise UsageLedgerCorrupt(f"usage archive path cannot be inspected: {level}") from exc
        try:  # the question early; anything else is UNKNOWN (typed), never a silent empty answer
            mode = os.stat(root / ARCHIVE_SEGMENT_DIR_REL).st_mode
        except FileNotFoundError:
            return frozenset()  # never compacted and no archive: nothing to anchor against
        except OSError as exc:
            raise UsageLedgerCorrupt(
                f"usage archive directory cannot be inspected: {root / ARCHIVE_SEGMENT_DIR_REL}"
            ) from exc
        if not stat.S_ISDIR(mode):
            raise UsageLedgerCorrupt(
                f"usage archive level is not our own directory: {root / ARCHIVE_SEGMENT_DIR_REL}"
            )
    # POSIX holds the archive directory handles for the WHOLE question: the
    # chain walk and the epoch anchor read one and the same directory, whatever
    # the path names by the time the anchor runs.
    fds = _archive_dir_fds(root, create=False) if _dir_fd_capable() else []
    dir_fd = fds[-1] if fds else None
    header = live_header
    chain: list = []
    segments: list = []
    seen: set = set()
    expected_epoch: Optional[int] = None
    try:
        while header is not None:
            archive_rel = str(header.get("archive_rel") or "")
            expected = str(header.get("source_sha256") or "")
            epoch = header.get("compaction_epoch")
            if not archive_rel or not expected or isinstance(epoch, bool) or not isinstance(
                epoch, int
            ) or epoch < 1:
                raise UsageLedgerCorrupt("usage baseline header lacks archive provenance")
            if expected_epoch is not None and epoch != expected_epoch:
                raise UsageLedgerCorrupt(
                    f"usage archive chain epoch break: expected {expected_epoch}, found {epoch}"
                )
            if archive_rel in seen:
                raise UsageLedgerCorrupt(f"usage archive segment cycle at {archive_rel}")
            seen.add(archive_rel)
            segment_ids, header = _load_segment(root, header, dir_fd)
            chain.append((archive_rel, expected))
            segments.append(segment_ids)
            expected_epoch = epoch - 1
            if header is None and expected_epoch != 0:
                raise UsageLedgerCorrupt(
                    f"usage archive chain ends before epoch {expected_epoch}"
                )
        _no_newer_archived_epoch(
            root, live_header, {pathlib.PurePosixPath(rel).name for rel in seen}, dir_fd
        )
    finally:
        _close_fds(fds)
    key = tuple(chain)
    cached = _CHAIN_UNION_CACHE.get(key)
    if cached is not None:
        return cached
    union = _union_segment_ids(segments)
    if len(_CHAIN_UNION_CACHE) >= _CHAIN_UNION_CACHE_MAX:
        _CHAIN_UNION_CACHE.clear()
    _CHAIN_UNION_CACHE[key] = union
    return union


def usage_attempt_recorded(
    root: pathlib.Path | str | None,
    attempt_id: str,
    live_ids: Optional[set] = None,
) -> bool:
    """Membership of ``attempt_id`` in the live replay ∪ archived segments.

    The join primitive for per-attempt history questions on a compacted
    ledger (CPL-5 reverse sweep: an id absent HERE — not merely absent from
    the live replay — is what "no attempt row" means)."""
    attempt_id = str(attempt_id or "")
    if not attempt_id:
        return False
    if live_ids is None:
        from ouroboros._usage_rows_memo import _memoized_final_rows

        rows, _, _, _ = _memoized_final_rows(pathlib.Path(_drive_root(root)))
        live_ids = {str(row.get("attempt_id") or "") for row in rows}
    if attempt_id in live_ids:
        return True
    return attempt_id in archived_attempt_ids(root)
