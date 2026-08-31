"""Durable append-only usage ledger: the substrate the accounting layer writes on.

ONE job, kept apart from policy: own the bytes. Cross-process locking, atomic
append + fsync, structural validation of every row and transition, and
quarantine of a torn tail. It knows what a well-formed ledger row IS; it has no
opinion about reservations, budgets, pricing, or projections — those live in
``usage_accounting``, which imports FROM here and is never imported BY here.

The seam is one-way by construction, so the monetary authority (the file) cannot
be corrupted by a change in accounting policy, and a locking or fsync fix cannot
silently alter what a reservation means.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import pathlib
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from ouroboros.utils import append_jsonl, replace_atomic, utc_now_iso

log = logging.getLogger(__name__)

LEDGER_REL = pathlib.Path("state/usage_attempts.jsonl")
QUARANTINE_REL = pathlib.Path("state/usage_attempts.quarantine.jsonl")
_TERMINAL = frozenset({"settled", "unresolved", "released"})
# The typed settle_reason of the ONE legal exit from `unresolved`: the abandoned-attempt
# reconciler's write-off, settling the row at its carried reservation upper bound
# (ouroboros/usage_reconcile.py). Any other post-terminal mutation stays corrupt.
UNRESOLVED_WRITEOFF_REASON = "abandoned_unresolved_writeoff"

__all__ = (
    "LEDGER_REL", "QUARANTINE_REL", "UsageAccountingError", "UsageLedgerCorrupt",
)


class UsageAccountingError(RuntimeError):
    """Base error for fail-closed accounting operations."""


class UsageLedgerCorrupt(UsageAccountingError):
    """Raised when durable history is structurally invalid."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_IDENTITY_FIELDS = (
    "candidate_raw_sha256", "candidate_raw_size_bytes",
    "candidate_context_sha256", "candidate_context_size_bytes",
)


def _validate_candidate_facts(row: Dict[str, Any], sequence: int) -> None:
    if "candidate_payload" in row:
        raise UsageLedgerCorrupt(f"mutable candidate payload in usage row seq={sequence}")
    present = "candidate_measurement_kind" in row or any(key in row for key in _CANDIDATE_IDENTITY_FIELDS)
    if not present:
        return  # pre-feature/legacy rows
    kind = row.get("candidate_measurement_kind")
    if kind not in {"canonical_json_v1", "opaque"}:
        raise UsageLedgerCorrupt(f"invalid candidate_measurement_kind in usage row seq={sequence}")
    if kind == "opaque":
        if any(row.get(key) is not None for key in _CANDIDATE_IDENTITY_FIELDS):
            raise UsageLedgerCorrupt(f"opaque candidate claims identity in usage row seq={sequence}")
    else:
        for key in ("candidate_raw_sha256", "candidate_context_sha256"):
            if not isinstance(row.get(key), str) or not _SHA256_RE.fullmatch(row[key]):
                raise UsageLedgerCorrupt(f"invalid {key} in usage row seq={sequence}")
        for key in ("candidate_raw_size_bytes", "candidate_context_size_bytes"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UsageLedgerCorrupt(f"invalid {key} in usage row seq={sequence}")
    context = row.get("physical_context")
    if context is not None:
        if not isinstance(context, dict):
            raise UsageLedgerCorrupt(f"invalid physical_context in usage row seq={sequence}")
        if context.get("profile") not in {"owner_max", "owner_low", "task_local_low"}:
            raise UsageLedgerCorrupt(f"invalid physical_context profile in usage row seq={sequence}")
        if context.get("rendered_mode") not in {"max", "low"} or context.get("measurement_basis") not in {
            "fresh_route_usage", "fresh_model_usage", "cold_estimate",
        }:
            raise UsageLedgerCorrupt(f"invalid physical_context mode/basis in usage row seq={sequence}")
        for key in ("target_total_tokens", "capacity_total_tokens"):
            value = context.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise UsageLedgerCorrupt(f"invalid physical_context {key} in usage row seq={sequence}")
        if not all(isinstance(context.get(key), bool) for key in ("context_target_miss", "automatic_pass_used")):
            raise UsageLedgerCorrupt(f"invalid physical_context flags in usage row seq={sequence}")
        if not all(isinstance(context.get(key), str) for key in ("route_fp", "round_id")):
            raise UsageLedgerCorrupt(f"invalid physical_context identity in usage row seq={sequence}")
    manifest_ref = row.get("candidate_manifest_ref")
    if manifest_ref is not None and (
        not isinstance(manifest_ref, dict)
        or manifest_ref.get("call_id") != row.get("attempt_id")
        or not _SHA256_RE.fullmatch(str(manifest_ref.get("sha256") or ""))
    ):
        raise UsageLedgerCorrupt(f"invalid candidate_manifest_ref in usage row seq={sequence}")


def _drive_root(value: pathlib.Path | str | None = None) -> pathlib.Path:
    if value is not None:
        if not isinstance(value, (str, pathlib.Path)):
            raise UsageAccountingError(f"invalid usage accounting drive root type: {type(value).__name__}")
        resolved = pathlib.Path(value)
        if not resolved.is_absolute():
            raise UsageAccountingError(f"usage accounting drive root must be absolute: {resolved}")
        return resolved
    configured = str(os.environ.get("OUROBOROS_DATA_DIR") or "").strip()
    if configured:
        resolved = pathlib.Path(configured)
        if not resolved.is_absolute():
            raise UsageAccountingError(f"OUROBOROS_DATA_DIR must be absolute for usage accounting: {resolved}")
        return resolved
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR)


@contextlib.contextmanager
def _named_lock(
    root: pathlib.Path,
    filename: str,
    *,
    timeout_sec: float,
    stale_sec: float,
) -> Iterator[None]:
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    path = root / "state" / filename
    fd = acquire_exclusive_file_lock(path, timeout_sec=timeout_sec, stale_sec=stale_sec)
    if fd is None:
        raise UsageAccountingError(f"usage accounting lock unavailable: {path}")
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


@contextlib.contextmanager
def _locked(root: pathlib.Path) -> Iterator[None]:
    # Operator fix 2026-07-23: 4.0s starves under a grown ledger (reserve_attempt
    # re-reads the whole usage_attempts.jsonl under this lock — ~0.5s hold at 20MB),
    # failing healthy tasks with UsageAccountingError at >=10 concurrent workers.
    # Waiting longer is always correct here; the transaction itself stays atomic.
    with _named_lock(root, "usage_attempts.lock", timeout_sec=45.0, stale_sec=90.0):
        yield


def _append_bytes_fsync(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short append to {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_atomic_fsync(path: pathlib.Path, payload: bytes) -> None:
    """Persist the exact snapshotted bytes without reopening the source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")
    fd: Optional[int] = None
    try:
        # Windows defaults low-level descriptors to text mode, which would
        # expand LF bytes and break the archive's immutable source hash.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(str(tmp), flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write to {tmp}")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        replace_atomic(tmp, path)
    except Exception:
        if fd is not None:
            os.close(fd)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _quarantine_tail(root: pathlib.Path, raw: bytes, offset: int, reason: str) -> None:
    ledger = root / LEDGER_REL
    row = {
        "ts": utc_now_iso(),
        "reason": reason,
        "source": str(ledger),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }
    _append_bytes_fsync(
        root / QUARANTINE_REL,
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )
    fd = os.open(str(ledger), os.O_RDWR)
    try:
        os.ftruncate(fd, offset)
        os.fsync(fd)
    finally:
        os.close(fd)
    log.error("Quarantined corrupt final usage-ledger row: %s", reason)
    try:
        append_jsonl(
            root / "logs" / "events.jsonl",
            {"type": "usage_ledger_tail_quarantined", "ts": utc_now_iso(), "reason": reason},
        )
    except Exception:
        log.exception("Failed to emit usage-ledger quarantine event")


def _validate_records(
    records: Sequence[Dict[str, Any]],
    *,
    start_seq: int = 1,
    states: Optional[Dict[str, str]] = None,
) -> None:
    """Validate row structure, dense sequence, and per-attempt transitions.

    ``start_seq``/``states`` are the ADDITIVE resume seam for incremental tail
    validation: a caller that already validated a prefix passes the next
    expected sequence number and the prefix's per-attempt last-state map (which
    is mutated in place as the tail validates). Defaults reproduce the historic
    whole-ledger behavior exactly.
    """
    states = {} if states is None else states
    expected = int(start_seq)
    for row in records:
        try:
            sequence = int(row.get("seq") or 0) if isinstance(row, dict) else 0
        except (TypeError, ValueError, OverflowError) as exc:
            raise UsageLedgerCorrupt(f"invalid usage ledger sequence at {expected}") from exc
        if not isinstance(row, dict) or sequence != expected:
            raise UsageLedgerCorrupt(f"usage ledger sequence mismatch at {expected}")
        expected += 1
        attempt_id = str(row.get("attempt_id") or "")
        state = str(row.get("state") or "")
        kind = str(row.get("kind") or "attempt")
        if not attempt_id or state not in {"reserved", "dispatched", *_TERMINAL}:
            raise UsageLedgerCorrupt(f"invalid usage ledger row seq={row.get('seq')}")
        _validate_candidate_facts(row, sequence)
        for numeric_field in (
            "cost_usd", "reservation_upper_bound_usd", "reservation_usd",
            "max_budget_usd", "global_limit_usd", "root_limit_usd",
        ):
            if row.get(numeric_field) is not None and _number(row.get(numeric_field)) is None:
                raise UsageLedgerCorrupt(f"invalid {numeric_field} in usage row seq={sequence}")
        for token_field in (
            "prompt_tokens", "completion_tokens", "cached_tokens",
            "cache_write_tokens", "ambiguous_call_count",
        ):
            if row.get(token_field) is None:
                continue
            try:
                value = int(row.get(token_field))
            except (TypeError, ValueError, OverflowError) as exc:
                raise UsageLedgerCorrupt(
                    f"invalid {token_field} in usage row seq={sequence}"
                ) from exc
            if value < 0 or isinstance(row.get(token_field), bool):
                raise UsageLedgerCorrupt(
                    f"invalid {token_field} in usage row seq={sequence}"
                )
        previous = states.get(attempt_id)
        if kind.startswith("legacy_") or kind in {"external_unmetered", "subscription_session"}:
            if previous is not None or state not in {"settled", "unresolved"}:
                raise UsageLedgerCorrupt(f"invalid legacy usage row seq={row.get('seq')}")
        elif previous is None:
            if state != "reserved":
                raise UsageLedgerCorrupt(f"attempt {attempt_id} did not begin reserved")
        elif previous == "reserved":
            if state not in {"dispatched", "released"}:
                raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
        elif previous == "dispatched":
            if state not in {"settled", "unresolved", "released"}:
                raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
            if state == "released" and not str(row.get("reason") or "").startswith(
                "before_dispatch_failed:"
            ):
                raise UsageLedgerCorrupt(
                    f"dispatched->released requires a typed pre-dispatch reason at seq={row.get('seq')}"
                )
        elif previous == "unresolved":
            # The write-off must settle AT the carried bound, finally: a cheaper
            # or non-final "write-off" is a fabricated discount on real spend.
            if (
                state != "settled"
                or str(row.get("settle_reason") or "") != UNRESOLVED_WRITEOFF_REASON
                or row.get("cost_final") is not True
                or _number(row.get("cost_usd")) is None
                or _number(row.get("reservation_upper_bound_usd")) is None
                or _number(row.get("cost_usd")) != _number(row.get("reservation_upper_bound_usd"))
            ):
                raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
        else:
            raise UsageLedgerCorrupt(f"attempt {attempt_id} changed after terminal state")
        states[attempt_id] = state


def _read_records_locked(root: pathlib.Path) -> list[Dict[str, Any]]:
    path = root / LEDGER_REL
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise UsageAccountingError(f"cannot read usage ledger: {exc}") from exc
    records: list[Dict[str, Any]] = []
    record_locations: list[Tuple[int, bytes]] = []
    chunks = data.splitlines(keepends=True)
    nonempty = [index for index, chunk in enumerate(chunks) if chunk.rstrip(b"\r\n")]
    last_nonempty = nonempty[-1] if nonempty else -1
    offset = 0
    for index, chunk in enumerate(chunks):
        raw = chunk.rstrip(b"\r\n")
        if not raw:
            offset += len(chunk)
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if index == last_nonempty:
                _quarantine_tail(root, chunk, offset, f"{type(exc).__name__}: {exc}")
                break
            raise UsageLedgerCorrupt(f"corrupt usage ledger row before tail: {index + 1}") from exc
        records.append(row)
        record_locations.append((offset, chunk))
        offset += len(chunk)
    try:
        _validate_records(records)
    except UsageLedgerCorrupt:
        # A final row can be valid JSON yet still be torn structurally (wrong
        # seq, illegal transition, missing fields). Preserve the validated
        # history exactly as for a JSON-torn tail; corruption before the final
        # row remains a hard failure.
        if not records or not record_locations:
            raise
        try:
            _validate_records(records[:-1])
        except UsageLedgerCorrupt:
            raise
        bad_offset, bad_chunk = record_locations[-1]
        _quarantine_tail(root, bad_chunk, bad_offset, "structurally invalid final ledger row")
        records.pop()
    return records


@dataclass
class LedgerResumeState:
    """Where a validated read of the ledger ended, for incremental resumption.

    Identity (``st_ino``/``st_dev``), extent (``size`` = byte offset after the
    last validated row) and ``st_mtime_ns`` fingerprint the file as it was read
    UNDER THE LOCK; ``row_count`` and the per-attempt last-``states`` map seed
    tail validation so transition rules hold across the resume boundary. A
    missing ledger is represented as ``st_ino/st_dev = -1`` with ``size = 0``;
    ``st_ino/st_dev = -2`` marks a deliberately NON-RESUMABLE fingerprint (the
    file's tail is not row-aligned), which no real inode ever matches, so every
    subsequent read stays a full replay.
    """

    st_ino: int
    st_dev: int
    size: int
    st_mtime_ns: int
    row_count: int
    states: Dict[str, str] = field(default_factory=dict)


def _ledger_resume_state(
    root: pathlib.Path, records: Sequence[Dict[str, Any]]
) -> LedgerResumeState:
    """Fingerprint the just-read ledger for incremental resumption.

    Must be called under the same held ledger lock as the read that produced
    ``records`` (writers append only under that lock, so the stat is consistent
    with the validated content — including any quarantine truncation the read
    itself performed)."""
    states = {str(row.get("attempt_id") or ""): str(row.get("state") or "") for row in records}
    path = root / LEDGER_REL
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return LedgerResumeState(-1, -1, 0, -1, len(records), states)
    if stat.st_size > 0:
        try:
            with open(path, "rb") as handle:
                handle.seek(stat.st_size - 1)
                terminated = handle.read(1) == b"\n"
        except OSError:
            terminated = False
        if not terminated:
            # A crash-torn final line that is still valid JSON parses in the full
            # read, but its end is NOT a row boundary: an append landing directly
            # onto it welds rows into one unparseable line (the #138 guard in
            # _append_rows_locked repairs the boundary before writing, but reads
            # before any repair — or after a foreign blind append — must not
            # resume from a mid-line offset). Refuse until the tail is row-aligned.
            return LedgerResumeState(-2, -2, stat.st_size, -1, len(records), states)
    return LedgerResumeState(
        stat.st_ino, stat.st_dev, stat.st_size, stat.st_mtime_ns, len(records), states
    )


def _read_new_records_locked(
    root: pathlib.Path, resume: LedgerResumeState
) -> Optional[Tuple[list[Dict[str, Any]], LedgerResumeState]]:
    """Incrementally read rows appended after ``resume``; ``None`` = full refold.

    Returns ``(new_records, new_resume)`` when the resume fingerprint still
    matches and the appended tail parses and validates as a seq-continuous,
    transition-legal continuation. Returns ``None`` whenever the resume state
    cannot be trusted — file replaced (inode/device change), shrunk below the
    resume offset, rewritten in place (same size, different mtime), or a
    torn/structurally invalid tail — so the caller re-reads through the normal
    ``_read_records_locked``, which OWNS quarantine. This function never
    truncates or otherwise mutates the ledger, and must be called under the
    held ledger lock.
    """
    path = root / LEDGER_REL
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        if resume.row_count == 0 and resume.size == 0:
            return [], resume
        return None
    except OSError:
        return None
    if (stat.st_ino, stat.st_dev) != (resume.st_ino, resume.st_dev):
        return None
    if stat.st_size < resume.size:
        return None
    if stat.st_size == resume.size:
        return ([], resume) if stat.st_mtime_ns == resume.st_mtime_ns else None
    try:
        with open(path, "rb") as handle:
            handle.seek(resume.size)
            data = handle.read()
    except OSError:
        return None
    if not data.endswith(b"\n"):
        # A torn in-flight append (crashed writer). The full reader decides
        # whether that tail is quarantined; never guess here.
        return None
    records: list[Dict[str, Any]] = []
    for chunk in data.splitlines():
        raw = chunk.rstrip(b"\r")
        if not raw:
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        records.append(row)
    seeded_states = dict(resume.states)
    try:
        _validate_records(records, start_seq=resume.row_count + 1, states=seeded_states)
    except UsageLedgerCorrupt:
        return None
    return records, LedgerResumeState(
        stat.st_ino,
        stat.st_dev,
        stat.st_size,
        stat.st_mtime_ns,
        resume.row_count + len(records),
        seeded_states,
    )


def _append_rows_locked(
    root: pathlib.Path,
    records: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if not rows:
        return []
    sequence = len(records)
    materialized: list[Dict[str, Any]] = []
    for raw in rows:
        sequence += 1
        materialized.append({**raw, "seq": sequence, "ts": str(raw.get("ts") or utc_now_iso())})
    _validate_records([*records, *materialized])
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in materialized
    )
    # razzant/ouroboros#138: O_APPEND writes payload verbatim after whatever is
    # already on disk. If a prior writer died mid-append it can have left a
    # newline-less partial tail; appending straight onto it glues the partial
    # and the first new row into one unparseable line, and _read_records_locked
    # would then quarantine BOTH. The validated-tail readers already refuse to
    # warm-resume from such a file; this guards the raw byte boundary on write
    # so a torn tail costs at most itself, never the row that follows.
    ledger_path = root / LEDGER_REL
    try:
        with open(ledger_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    payload = b"\n" + payload
    except FileNotFoundError:
        pass
    _append_bytes_fsync(ledger_path, payload)
    return materialized


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


def _final_rows(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["attempt_id"]): row for row in records}
