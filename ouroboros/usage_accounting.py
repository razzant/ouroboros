"""Durable physical-model-attempt accounting.

The append-only JSONL ledger is the monetary authority; ``llm_usage`` and
``state.json`` remain compatibility projections carrying attempt ids. Ledger
validation and fsync share a short lock, while network I/O remains outside it.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import json
import logging
import os
import pathlib
import threading
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

from ouroboros.pricing import estimate_cost_optional
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

LEDGER_REL = pathlib.Path("state/usage_attempts.jsonl")
QUARANTINE_REL = pathlib.Path("state/usage_attempts.quarantine.jsonl")
IMPORT_REL = pathlib.Path("state/usage_import_watermark.json")
_TERMINAL = frozenset({"settled", "unresolved", "released"})

__all__ = (
    "AttemptRequest", "AttemptReservation", "BudgetExceeded", "PhysicalAttemptCapture",
    "PhysicalAttemptLimitExceeded",
    "UsageAccountingError", "UsageLedgerCorrupt", "UsageScope", "capture_attempt_ids",
    "current_usage_scope",
    "ensure_legacy_imported", "execute_physical_attempt", "execute_physical_attempt_async",
    "mark_dispatched", "mark_unresolved", "physical_attempt_limit",
    "physical_attempt_capture_from_exception", "physical_provider_outcome_unknown",
    "record_unmetered_external_dispatch", "release_attempt", "reserve_attempt", "settle_attempt",
    "usage_breakdown", "usage_from_response", "usage_projection", "usage_scope",
    "review_wave_admission",
)
_CURRENT_SCOPE: contextvars.ContextVar[Optional["UsageScope"]] = contextvars.ContextVar(
    "ouroboros_usage_scope", default=None
)
_ATTEMPT_COLLECTOR: contextvars.ContextVar[Optional[list[str]]] = contextvars.ContextVar(
    "ouroboros_usage_attempt_collector", default=None
)
_PHYSICAL_LIMIT: contextvars.ContextVar[Optional["_AttemptLimit"]] = contextvars.ContextVar(
    "ouroboros_physical_attempt_limit", default=None
)


class UsageAccountingError(RuntimeError):
    """Base error for fail-closed accounting operations."""


class UsageLedgerCorrupt(UsageAccountingError):
    """Raised when durable history is structurally invalid."""


class BudgetExceeded(UsageAccountingError):
    """Raised before dispatch when a known budget would be exceeded."""

    def __init__(self, message: str, *, limit_scope: str = "global", root_task_id: str = "") -> None:
        super().__init__(message)
        self.limit_scope = str(limit_scope or "global")
        self.root_task_id = str(root_task_id or "")


class PhysicalAttemptLimitExceeded(UsageAccountingError):
    """Raised before a provider send would exceed the caller's actor-local rail."""


@dataclass
class _AttemptLimit:
    maximum: int
    used: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class UsageScope:
    drive_root: pathlib.Path | str | None = None
    task_id: str = ""
    root_task_id: str = ""
    parent_task_id: str = ""
    category: str = "task"
    source: str = "llm"
    global_limit_usd: Optional[float] = None
    root_limit_usd: Optional[float] = None


@dataclass(frozen=True)
class AttemptRequest:
    model: str
    provider: str
    prompt_tokens_estimate: int = 0
    max_completion_tokens: int = 0
    reservation_usd: Optional[float] = None
    max_budget_usd: Optional[float] = None
    global_limit_usd: Optional[float] = None
    drive_root: pathlib.Path | str | None = None
    task_id: str = ""
    root_task_id: str = ""
    parent_task_id: str = ""
    category: str = ""
    source: str = ""
    root_limit_usd: Optional[float] = None
    force_unknown_reservation: bool = False


@dataclass(frozen=True)
class AttemptReservation:
    attempt_id: str
    drive_root: pathlib.Path
    model: str
    provider: str
    reservation_upper_bound_usd: Optional[float]


@dataclass(frozen=True)
class PhysicalAttemptCapture:
    """Immutable ledger and provider-response facts for one raised send."""
    attempt_id: str
    model: str
    provider: str
    state: str
    provider_response_observed: bool = False
    provider_status_code: Optional[int] = None
    provider_code: str = ""
    logical_call_attempt_ids: Tuple[str, ...] = ()


def physical_attempt_capture_from_exception(exc: BaseException) -> Optional[PhysicalAttemptCapture]:
    """Read direct/causal capture; ignore ambiguous automatic ``__context__``."""
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        capture = getattr(current, "physical_attempt_capture", None)
        if isinstance(capture, PhysicalAttemptCapture):
            return capture
        current = current.__cause__
    return None


def physical_provider_outcome_unknown(exc: BaseException) -> bool:
    """Whether replaying this exact dispatched request could duplicate work."""
    capture = physical_attempt_capture_from_exception(exc)
    return bool(capture is not None
                and capture.state in {"dispatched", "unresolved"}
                and not capture.provider_response_observed)


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
def usage_scope(scope: UsageScope) -> Iterator[UsageScope]:
    """Bind task/root attribution for physical sends in this execution context."""
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


def current_usage_scope() -> Optional[UsageScope]:
    """Return the immutable scope bound to this execution context, if any."""
    return _CURRENT_SCOPE.get()


@contextlib.contextmanager
def capture_attempt_ids() -> Iterator[list[str]]:
    """Collect physical attempt ids for one compatibility ``llm_usage`` row."""
    bucket: list[str] = []
    token = _ATTEMPT_COLLECTOR.set(bucket)
    try:
        yield bucket
    finally:
        _ATTEMPT_COLLECTOR.reset(token)


@contextlib.contextmanager
def physical_attempt_limit(maximum: int) -> Iterator[None]:
    """Bound physical provider sends in this actor context (acceptance uses 2)."""
    state = _AttemptLimit(maximum=max(0, int(maximum)))
    token = _PHYSICAL_LIMIT.set(state)
    try:
        yield
    finally:
        _PHYSICAL_LIMIT.reset(token)


def _claim_physical_dispatch() -> None:
    state = _PHYSICAL_LIMIT.get()
    if state is None:
        return
    with state.lock:
        if state.used >= state.maximum:
            raise PhysicalAttemptLimitExceeded(f"physical attempt limit exhausted ({state.used}/{state.maximum})")
        state.used += 1


def _merge_scope(request: AttemptRequest) -> Tuple[AttemptRequest, UsageScope]:
    bound = _CURRENT_SCOPE.get() or UsageScope()
    scope = UsageScope(
        drive_root=request.drive_root or bound.drive_root,
        task_id=str(request.task_id or bound.task_id or ""),
        root_task_id=str(request.root_task_id or bound.root_task_id or ""),
        parent_task_id=str(request.parent_task_id or bound.parent_task_id or ""),
        category=str(request.category or bound.category or "task"),
        source=str(request.source or bound.source or "llm"),
        global_limit_usd=(
            request.global_limit_usd if request.global_limit_usd is not None else bound.global_limit_usd
        ),
        root_limit_usd=(request.root_limit_usd if request.root_limit_usd is not None else bound.root_limit_usd),
    )
    if not scope.root_task_id and scope.task_id:
        scope = replace(scope, root_task_id=scope.task_id)
    if request.global_limit_usd is None and scope.global_limit_usd is not None:
        request = replace(request, global_limit_usd=scope.global_limit_usd)
    return request, scope


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
        os.replace(tmp, path)
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


def _validate_records(records: Sequence[Dict[str, Any]]) -> None:
    states: Dict[str, str] = {}
    expected = 1
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
        if kind.startswith("legacy_") or kind == "external_unmetered":
            if previous is not None or state not in {"settled", "unresolved"}:
                raise UsageLedgerCorrupt(f"invalid legacy usage row seq={row.get('seq')}")
        elif previous is None:
            if state != "reserved":
                raise UsageLedgerCorrupt(f"attempt {attempt_id} did not begin reserved")
        elif previous == "reserved":
            if state not in {"dispatched", "released"}:
                raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
        elif previous == "dispatched":
            if state not in {"settled", "unresolved"}:
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
    _append_bytes_fsync(root / LEDGER_REL, payload)
    return materialized


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


def _final_rows(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["attempt_id"]): row for row in records}


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    settled = confirmed = estimated = reserved = unresolved = 0.0
    unknown = 0
    counts: Dict[str, int] = {}
    for row in rows:
        state = str(row.get("state") or "")
        if str(row.get("kind") or "") == "legacy_metadata":
            ambiguous = max(1, int(row.get("ambiguous_call_count") or 1))
            counts["metadata_only"] = counts.get("metadata_only", 0) + ambiguous
            continue
        counts[state] = counts.get(state, 0) + 1
        pricing_unknown = row.get("pricing_known") is False
        if state == "settled":
            cost = _number(row.get("cost_usd"))
            if cost is None:
                unknown += 1
                bound = _number(row.get("reservation_upper_bound_usd"))
                if bound is not None:
                    unresolved += bound
            else:
                settled += cost
                if bool(row.get("cost_final")):
                    confirmed += cost
                else:
                    estimated += cost
        elif state == "reserved":
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                unknown += 1
            if bound is not None:
                reserved += bound
        elif state in {"dispatched", "unresolved"}:
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                unknown += 1
            if bound is not None:
                unresolved += bound
    settled, confirmed, estimated, reserved, unresolved = (
        round(value, 6) for value in (settled, confirmed, estimated, reserved, unresolved)
    )
    return {
        "settled_usd": settled,
        "confirmed_usd": confirmed,
        "estimated_usd": estimated,
        "reserved_usd": reserved,
        "unresolved_upper_bound_usd": unresolved,
        "accounted_usd": round(settled + reserved + unresolved, 6),
        "unknown_unmetered": unknown,
        "cost_final": not unknown and not reserved and not unresolved and not estimated,
        "attempt_counts": counts,
    }


def _with_limit(summary: Dict[str, Any], limit: Optional[float]) -> Dict[str, Any]:
    if limit is None:
        return summary
    summary["limit_usd"] = round(max(0.0, float(limit)), 6)
    summary["remaining_known_usd"] = round(max(0.0, summary["limit_usd"] - float(summary["accounted_usd"])), 6)
    return summary


def _with_integrity(summary: Dict[str, Any], degraded: bool) -> Dict[str, Any]:
    """Attach ledger integrity and prevent a torn tail from claiming final cost."""
    summary["integrity_degraded"] = bool(degraded)
    if degraded:
        summary["cost_final"] = False
    return summary


def usage_projection(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    global_limit_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a replayed global projection, or one root/subtree projection."""
    root = _drive_root(drive_root)
    with _locked(root):
        final = list(_final_rows(_read_records_locked(root)).values())
    integrity_degraded = (root / QUARANTINE_REL).is_file()
    if root_task_id:
        final = [row for row in final if str(row.get("root_task_id") or "") == root_task_id]
        limits = [_number(row.get("root_limit_usd")) for row in final]
        known_limits = [value for value in limits if value is not None]
        result = _with_limit(_summary(final), min(known_limits) if known_limits else None)
        return _with_integrity(result, integrity_degraded)
    if global_limit_usd is not None:
        configured_limit = max(0.0, float(global_limit_usd))
    else:
        try:
            configured_limit = float(os.environ.get("TOTAL_BUDGET", "10") or 0.0)
        except (TypeError, ValueError):
            configured_limit = 10.0
    result = (
        _with_limit(_summary(final), configured_limit)
        if global_limit_usd is not None or configured_limit > 0
        else _summary(final)
    )
    root_ids = sorted({str(row.get("root_task_id") or "") for row in final if row.get("root_task_id")})
    result["by_root"] = {}
    for rid in root_ids:
        root_rows = [row for row in final if str(row.get("root_task_id") or "") == rid]
        known_limits = [
            value for value in (_number(row.get("root_limit_usd")) for row in root_rows) if value is not None
        ]
        result["by_root"][rid] = _with_integrity(
            _with_limit(_summary(root_rows), min(known_limits) if known_limits else None),
            integrity_degraded,
        )
    return _with_integrity(result, integrity_degraded)


def _physical_call_count(row: Dict[str, Any]) -> int:
    kind = str(row.get("kind") or "attempt")
    if kind == "legacy_metadata":
        return 0
    if kind == "legacy_delta":
        return 0
    return 1 if str(row.get("state") or "") in {"dispatched", "settled", "unresolved"} else 0


def _breakdown_bucket(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bucket = _summary(rows)
    prompt = sum(max(0, int(row.get("prompt_tokens") or 0)) for row in rows)
    completion = sum(max(0, int(row.get("completion_tokens") or 0)) for row in rows)
    prompt_cache_ttls: Dict[str, int] = {}
    for row in rows:
        ttl = str(row.get("prompt_cache_ttl") or "").strip()
        if ttl:
            prompt_cache_ttls[ttl] = prompt_cache_ttls.get(ttl, 0) + _physical_call_count(row)
    bucket.update({
        "physical_calls": sum(_physical_call_count(row) for row in rows),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cached_tokens": sum(max(0, int(row.get("cached_tokens") or 0)) for row in rows),
        "cache_write_tokens": sum(max(0, int(row.get("cache_write_tokens") or 0)) for row in rows),
        "prompt_cache_ttls": prompt_cache_ttls,
    })
    return bucket


def usage_breakdown(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    task_id: str = "",
) -> Dict[str, Any]:
    """Read-only physical-call/token/cost buckets from validated ledger finals."""
    root = _drive_root(drive_root)
    with _locked(root):
        rows = list(_final_rows(_read_records_locked(root)).values())
    integrity_degraded = (root / QUARANTINE_REL).is_file()
    if root_task_id:
        rows = [row for row in rows if str(row.get("root_task_id") or "") == root_task_id]
    if task_id:
        rows = [row for row in rows if str(row.get("task_id") or "") == task_id]

    def grouped(field: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        groups: Dict[str, list[Dict[str, Any]]] = {}
        unattributed: list[Dict[str, Any]] = []
        for row in rows:
            key = str(row.get(field) or "")
            if str(row.get("kind") or "") in {"legacy_metadata", "legacy_delta"} or not key:
                unattributed.append(row)
            else:
                groups.setdefault(key, []).append(row)
        return (
            {key: _breakdown_bucket(groups[key]) for key in sorted(groups)},
            _breakdown_bucket(unattributed),
        )

    by_model, model_unattributed = grouped("model")
    by_provider, provider_unattributed = grouped("provider")
    by_category, category_unattributed = grouped("category")
    by_task, task_unattributed = grouped("task_id")
    by_root, root_unattributed = grouped("root_task_id")

    result = {
        **_with_integrity(_breakdown_bucket(rows), integrity_degraded),
        "by_model": by_model,
        "by_provider": by_provider,
        "by_category": by_category,
        "by_task": by_task,
        "by_root": by_root,
        # Legacy call-count metadata and monetary delta stay explicit; neither
        # is fabricated into a model/provider/category identity.
        "unattributed": {
            "model": model_unattributed,
            "provider": provider_unattributed,
            "category": category_unattributed,
            "task": task_unattributed,
            "root": root_unattributed,
        },
    }
    if integrity_degraded:
        for grouped_buckets in (
            by_model, by_provider, by_category, by_task, by_root,
            result["unattributed"],
        ):
            for bucket in grouped_buckets.values():
                _with_integrity(bucket, True)
    return result


def _reservation_cost(request: AttemptRequest) -> Optional[float]:
    explicit = request.max_budget_usd if request.max_budget_usd is not None else request.reservation_usd
    if explicit is not None:
        return _number(explicit)
    if request.force_unknown_reservation:
        return None
    if str(request.provider or "").lower() == "local":
        return 0.0
    prompt_tokens = max(0, int(request.prompt_tokens_estimate or 0))
    # chars/4 is a planning estimate, not a safe monetary hold.  OpenAI-family
    # tokenization on the live v6.64 smoke measured 1.0382x that estimate; keep
    # a provider-specific 1.10 linear envelope so the reservation also selects
    # the correct long-context price tier.  Explicit/opaque caps and settlement
    # remain untouched above/below this estimator.
    from ouroboros.provider_models import normalize_model_identity

    normalized_model = normalize_model_identity(str(request.model or "").lstrip("~"))
    if (
        str(request.provider or "").strip().lower() in {"openai", "openrouter"}
        and normalized_model.startswith("openai/")
    ):
        prompt_tokens = (prompt_tokens * 11 + 9) // 10
    cache_write_tokens = (
        prompt_tokens if str(request.model or "").lstrip("~").startswith(("anthropic/", "anthropic::")) else 0
    )
    return estimate_cost_optional(
        request.model,
        prompt_tokens,
        max(0, int(request.max_completion_tokens or 0)),
        cache_write_tokens=cache_write_tokens,
        prompt_cache_ttl="1h" if cache_write_tokens else None,
        allow_live_fetch=True,
        provider=request.provider,
    )


def review_wave_admission(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str,
    models: Sequence[str],
    prompt_chars: int,
    max_completion_tokens: int = 65536,
) -> Dict[str, Any]:
    """Read-only pre-flight: does a full review wave fit the remaining root budget?

    A review wave (one reviewer call per slot) that cannot possibly fit dies
    mid-wave today: the first N slots spend real money, a later slot hits the
    root-budget gate, the review hangs ``pending``, and the task dies
    ``budget_exhausted`` with paid-but-useless partial review work. This helper
    lets a review surface decline the WHOLE wave up front and finalize honestly
    instead. It reuses the exact per-attempt reservation math
    (``_reservation_cost``) and the ledger projection — no second budget
    authority. Fail-open by design: no root scope, no known limit, or unknown
    pricing for any slot → ``fits=True`` (identical to the admission stance in
    ``reserve_attempt``, where unknown price never blocks dispatch)."""
    result: Dict[str, Any] = {
        "fits": True,
        "estimated_wave_usd": None,
        "remaining_usd": None,
        "limit_usd": None,
        "slots": len(list(models or [])),
    }
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id or not models:
        return result
    try:
        from ouroboros.pricing import infer_provider_from_model

        projection = usage_projection(drive_root, root_task_id=root_task_id)
        limit = _number(projection.get("limit_usd"))
        remaining = _number(projection.get("remaining_known_usd"))
        if limit is None or remaining is None:
            return result
        result["limit_usd"] = limit
        result["remaining_usd"] = remaining
        prompt_tokens = max(0, int(prompt_chars or 0)) // 4
        total = 0.0
        for model in models:
            bound = _reservation_cost(
                AttemptRequest(
                    model=str(model or ""),
                    provider=infer_provider_from_model(str(model or "")),
                    prompt_tokens_estimate=prompt_tokens,
                    max_completion_tokens=max(0, int(max_completion_tokens or 0)),
                )
            )
            if bound is None:
                return result
            total += float(bound)
        result["estimated_wave_usd"] = round(total, 6)
        result["fits"] = total <= remaining + 1e-9
        return result
    except Exception:
        log.debug("review_wave_admission failed open", exc_info=True)
        return result


def _global_limit(request: AttemptRequest) -> float:
    if request.global_limit_usd is not None:
        return max(0.0, float(request.global_limit_usd))
    try:
        configured = float(os.environ.get("TOTAL_BUDGET", "10") or 0.0)
        return configured if configured > 0 else float("inf")
    except (TypeError, ValueError):
        return 10.0


def _active_root_budget_fence(root: pathlib.Path, root_task_id: str) -> Optional[Dict[str, Any]]:
    """Read the queue's atomic durable root-dispatch fence, if present."""
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return None
    snapshot_path = root / "state" / "queue_snapshot.json"
    if not snapshot_path.exists():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UsageAccountingError(
            f"root budget fence authority unavailable: {snapshot_path}"
        ) from exc
    rows = snapshot.get("budget_root_fences", []) if isinstance(snapshot, dict) else None
    if not isinstance(rows, list):
        raise UsageAccountingError(f"invalid root budget fence authority: {snapshot_path}")
    for row in rows:
        if not isinstance(row, dict):
            raise UsageAccountingError(f"invalid root budget fence row: {snapshot_path}")
        if (
            str(row.get("root_task_id") or "") == root_task_id
            and str(row.get("status") or "") in {"active", "paused"}
        ):
            return row
    return None


def reserve_attempt(request: AttemptRequest) -> AttemptReservation:
    """Atomically check global/root limits and append a ``reserved`` record."""
    request, scope = _merge_scope(request)
    root = _drive_root(scope.drive_root)
    root_fence = _active_root_budget_fence(root, scope.root_task_id)
    if root_fence is not None:
        raise BudgetExceeded(
            f"root model dispatch paused pending explicit resume for {scope.root_task_id}",
            limit_scope="root",
            root_task_id=scope.root_task_id,
        )
    ensure_legacy_imported(root)
    # IMPORTANT: live catalog I/O belongs before ``with _locked(root)`` below.
    # The lock protects only the atomic budget read/check/append transaction.
    bound = _reservation_cost(request)
    pricing_known = bound is not None
    attempt_id = uuid.uuid4().hex
    with _locked(root):
        records = _read_records_locked(root)
        finals = list(_final_rows(records).values())
        global_summary = _summary(finals)
        global_limit = _global_limit(request)
        accounted = float(global_summary["accounted_usd"])
        if global_limit <= 0 or accounted >= global_limit - 1e-9 or (
            bound is not None and accounted + bound > global_limit + 1e-9
        ):
            raise BudgetExceeded(
                f"global model budget exhausted: accounted=${accounted:.6f}, "
                f"reservation={'unknown' if bound is None else f'${bound:.6f}'}, limit=${global_limit:.6f}",
                limit_scope="global",
                root_task_id=scope.root_task_id,
            )
        if scope.root_task_id and scope.root_limit_usd is not None:
            root_rows = [row for row in finals if str(row.get("root_task_id") or "") == scope.root_task_id]
            root_accounted = float(_summary(root_rows)["accounted_usd"])
            root_limit = max(0.0, float(scope.root_limit_usd))
            if root_limit <= 0 or root_accounted >= root_limit - 1e-9 or (
                bound is not None and root_accounted + bound > root_limit + 1e-9
            ):
                raise BudgetExceeded(
                    f"root model budget exhausted for {scope.root_task_id}: "
                    f"accounted=${root_accounted:.6f}, limit=${root_limit:.6f}",
                    limit_scope="root",
                    root_task_id=scope.root_task_id,
                )
        _append_rows_locked(
            root,
            records,
            [
                {
                    "kind": "attempt",
                    "attempt_id": attempt_id,
                    "state": "reserved",
                    "model": str(request.model or ""),
                    "provider": str(request.provider or "unknown"),
                    "reservation_upper_bound_usd": bound,
                    "pricing_known": pricing_known,
                    "reservation_basis": (
                        "opaque_unknown"
                        if request.force_unknown_reservation and bound is None
                        else ("unknown_pricing" if not pricing_known
                        else ("explicit_upper_bound" if request.max_budget_usd is not None else "linear_pricing")
                        )
                    ),
                    "task_id": scope.task_id,
                    "root_task_id": scope.root_task_id,
                    "parent_task_id": scope.parent_task_id,
                    "category": scope.category,
                    "source": scope.source,
                    "global_limit_usd": request.global_limit_usd,
                    "root_limit_usd": scope.root_limit_usd,
                }
            ],
        )
    bucket = _ATTEMPT_COLLECTOR.get()
    if bucket is not None:
        bucket.append(attempt_id)
    return AttemptReservation(attempt_id, root, request.model, request.provider, bound)


def record_unmetered_external_dispatch(
    dispatch_id: str,
    *,
    drive_root: pathlib.Path | str | None = None,
    model: str = "",
    provider: str = "external",
    task_id: str = "",
    root_task_id: str = "",
    parent_task_id: str = "",
    category: str = "external",
    source: str = "external_skill",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> str:
    """Idempotently record a dispatch whose transport bypasses core metering."""
    stable_id = str(dispatch_id or "").strip()
    if not stable_id:
        raise UsageAccountingError("external unmetered dispatch requires a stable dispatch_id")
    bound = _CURRENT_SCOPE.get() or UsageScope()
    root = _drive_root(drive_root or bound.drive_root)
    ensure_legacy_imported(root)
    identity = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    attempt_id = f"external-{identity[:24]}"
    row = {
        "kind": "external_unmetered",
        "attempt_id": attempt_id,
        "state": "settled",
        "model": str(model or ""),
        "provider": str(provider or "external"),
        "cost_usd": None,
        "cost_final": False,
        "reservation_upper_bound_usd": None,
        "prompt_tokens": max(0, int(prompt_tokens or 0)),
        "completion_tokens": max(0, int(completion_tokens or 0)),
        "task_id": str(task_id or bound.task_id or ""),
        "root_task_id": str(root_task_id or bound.root_task_id or task_id or bound.task_id or ""),
        "parent_task_id": str(parent_task_id or bound.parent_task_id or ""),
        "category": str(category or bound.category or "external"),
        "source": str(source or bound.source or "external_skill"),
        "external_dispatch_id_sha256": identity,
    }
    with _locked(root):
        records = _read_records_locked(root)
        existing = _final_rows(records).get(attempt_id)
        if existing is not None:
            comparable = (
                "kind", "model", "provider", "task_id", "root_task_id", "parent_task_id",
                "category", "source", "prompt_tokens", "completion_tokens",
                "external_dispatch_id_sha256",
            )
            if any(existing.get(key) != row.get(key) for key in comparable):
                raise UsageAccountingError(f"conflicting external dispatch identity: {attempt_id}")
            return attempt_id
        _append_rows_locked(root, records, [row])
    return attempt_id


def _transition(reservation: AttemptReservation, state: str, **fields: Any) -> Dict[str, Any]:
    with _locked(reservation.drive_root):
        records = _read_records_locked(reservation.drive_root)
        current = _final_rows(records).get(reservation.attempt_id)
        if current is None:
            raise UsageAccountingError(f"unknown usage attempt {reservation.attempt_id}")
        row = {
            "kind": "attempt",
            "attempt_id": reservation.attempt_id,
            "state": state,
            "model": reservation.model,
            "provider": reservation.provider,
            "reservation_upper_bound_usd": reservation.reservation_upper_bound_usd,
            "pricing_known": current.get("pricing_known"),
            "reservation_basis": current.get("reservation_basis"),
            "task_id": str(current.get("task_id") or ""),
            "root_task_id": str(current.get("root_task_id") or ""),
            "parent_task_id": str(current.get("parent_task_id") or ""),
            "category": str(current.get("category") or "task"),
            "source": str(current.get("source") or "llm"),
            "global_limit_usd": current.get("global_limit_usd"),
            "root_limit_usd": current.get("root_limit_usd"),
            **fields,
        }
        return _append_rows_locked(reservation.drive_root, records, [row])[0]


def mark_dispatched(reservation: AttemptReservation) -> None:
    try:
        _claim_physical_dispatch()
    except PhysicalAttemptLimitExceeded:
        release_attempt(reservation, "physical_attempt_limit")
        raise
    _transition(reservation, "dispatched")


def release_attempt(reservation: AttemptReservation, reason: str = "not_dispatched") -> None:
    _transition(reservation, "released", reason=str(reason or "not_dispatched"))


def mark_unresolved(reservation: AttemptReservation, reason: str) -> None:
    _transition(reservation, "unresolved", reason=str(reason or "provider_outcome_unknown")[:500])


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    return value


def usage_from_response(response: Any) -> Tuple[Dict[str, Any], Optional[float], bool]:
    """Extract common provider usage/cost fields without persisting response text."""
    payload: Any = _plain(response)
    if not isinstance(payload, dict) and hasattr(response, "json"):
        try:
            payload = response.json()
        except Exception:
            payload = None
    usage: Any = payload.get("usage") if isinstance(payload, dict) else getattr(response, "usage", None)
    usage = _plain(usage)
    if not isinstance(usage, dict):
        usage = {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    normalized = {
        **usage,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }
    # An OpenAI-compatible HTTP-200 whose body is a top-level provider error
    # (OpenRouter passes 429/5xx/4xx through the body) that billed zero tokens is
    # a request rejected BEFORE generation — nothing was charged. Settle a
    # confirmed $0 so the attempt's conservative reservation upper bound is
    # RELEASED instead of held as an unresolved "phantom" that accumulates across
    # a provider storm and exhausts the task/global budget while real spend stays
    # small (v6.64.0 physical-attempt ledger blind spot; SWE-Pro tasks died
    # budget_exhausted at ~$7.5 of a $25 cap). The reservation stays held for the
    # cases that must not under-count: any billed tokens (prompt/completion > 0 —
    # a partial stream or an unknown-price success) fall through to the normal
    # cost path and keep their bound; the SDK/opaque-session path settles via
    # mark_unresolved and is untouched.
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and prompt == 0
        and completion == 0
    ):
        return normalized, 0.0, True
    cost_value = None
    for candidate in (
        usage.get("cost"),
        usage.get("total_cost"),
        payload.get("total_cost_usd") if isinstance(payload, dict) else None,
        getattr(response, "total_cost_usd", None),
    ):
        if candidate is not None:
            cost_value = candidate
            break
    cost = _number(cost_value)
    return normalized, cost, cost is not None


def settle_attempt(
    reservation: AttemptReservation,
    usage: Optional[Dict[str, Any]] = None,
    *,
    cost_usd: Optional[float] = None,
    cost_final: bool = False,
) -> None:
    normalized = dict(usage or {})
    cost = _number(cost_usd)
    has_usage = bool(
        int(normalized.get("prompt_tokens") or normalized.get("input_tokens") or 0)
        or int(normalized.get("completion_tokens") or normalized.get("output_tokens") or 0)
    )
    if cost is None and str(reservation.provider or "").lower() == "local":
        cost, cost_final = 0.0, True
    elif cost is None and has_usage:
        cost = estimate_cost_optional(
            reservation.model,
            int(normalized.get("prompt_tokens") or normalized.get("input_tokens") or 0),
            int(normalized.get("completion_tokens") or normalized.get("output_tokens") or 0),
            cached_tokens=int(normalized.get("cached_tokens") or 0),
            cache_write_tokens=int(normalized.get("cache_write_tokens") or 0),
            prompt_cache_ttl=str(normalized.get("prompt_cache_ttl") or ""),
            allow_live_fetch=False,
            provider=reservation.provider,
        )
        cost_final = False
    _transition(
        reservation,
        "settled",
        cost_usd=cost,
        cost_final=bool(cost_final and cost is not None),
        prompt_tokens=int(normalized.get("prompt_tokens") or normalized.get("input_tokens") or 0),
        completion_tokens=int(normalized.get("completion_tokens") or normalized.get("output_tokens") or 0),
        cached_tokens=int(normalized.get("cached_tokens") or 0),
        cache_write_tokens=int(normalized.get("cache_write_tokens") or 0),
        prompt_cache_ttl=str(normalized.get("prompt_cache_ttl") or ""),
    )


# Providers whose reported ``prompt_tokens`` INCLUDES cache reads and cache writes —
# the assumption `pricing.py` already encodes when it derives
# `regular_input = prompt_tokens - cached - cache_write`. On these routes a cached call
# still measures density correctly, so cache markers must NOT suppress the observation:
# the main loop and every review surface mark a stable prefix, so a
# skip-on-any-cache-token rule would silently make the measurement path VACUOUS and
# freeze every pack at the conservative cold-start density forever.
# `anthropic` belongs here as of v6.81.0: the direct-Anthropic path now reports
# `prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
# (`llm.py`, v6.77.0), i.e. the same OpenAI-semantics total the rest of this set has.
# It was correctly ABSENT while that path still reported bare `input_tokens`; leaving it
# out afterwards made the measurement vacuous on the main and heavy slots, which are the
# direct-Anthropic routes. GigaChat stays out: its `precached_prompt_tokens` semantics
# are still undocumented.
_CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS = frozenset({
    "openrouter", "openai", "openai-compatible", "cloudru", "local", "anthropic",
})


def _observe_token_density(request: AttemptRequest, usage: Optional[Dict[str, Any]]) -> None:
    """Learn the model's real tokenizer density from one settled physical send.

    Called AFTER settlement and therefore OUTSIDE the ledger lock, so a
    capability-evidence write can never delay or starve monetary accounting; every
    failure is swallowed (fail-soft — an unlearned observation only means review
    packs keep using the conservative cold-start density).

    A cache-bearing send is skipped ONLY on a route whose ``prompt_tokens`` semantics
    are not known to be cache-inclusive — today GigaChat, whose `precached_prompt_tokens`
    semantics are undocumented. There a partially cached call would report a falsely LOW
    density, and an under-measured density LOOSENS the review-pack cap — the one direction
    that reintroduces provider 400s. No arithmetic reconstruction is attempted here on
    purpose; the direct-Anthropic path needs none, because as of v6.77.0 it already folds
    cache reads and writes into ``prompt_tokens`` itself."""
    try:
        normalized = dict(usage or {})
        cache_bearing = bool(
            int(normalized.get("cached_tokens") or 0)
            or int(normalized.get("cache_write_tokens") or 0)
        )
        provider = str(request.provider or "").strip().lower()
        if cache_bearing and provider not in _CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS:
            return
        real = int(normalized.get("prompt_tokens") or normalized.get("input_tokens") or 0)
        estimate = int(request.prompt_tokens_estimate or 0)
        if real <= 0 or estimate <= 0:
            return
        from ouroboros.capability_evidence import record_token_density
        from ouroboros.provider_models import normalize_model_identity

        record_token_density(
            _drive_root(request.drive_root),
            normalize_model_identity(str(request.model or "")),
            prompt_chars=estimate * 4,
            prompt_tokens=real,
            source="dispatch_usage",
        )
    except Exception:
        log.debug("token-density observation skipped", exc_info=True)


def _is_pre_routing_rejection(exc: BaseException) -> bool:
    """True for an OpenRouter router-side 404 that never reached an upstream.

    OpenRouter answers "No endpoints found ..." (e.g. unsupported request
    parameters under require_parameters) with HTTP 404 from its ROUTER — no
    provider generation happened and nothing was billed. Settling such an
    attempt at a confirmed $0 releases its conservative reservation instead of
    holding a phantom upper bound against the root budget forever (the same
    blind-spot class as the v6.65.4 zero-usage body-error fix). Deliberately
    narrow: BOTH the 404 status and the router signature are required (and the
    caller additionally gates on provider == "openrouter"), so ordinary
    provider 404/5xx/timeout failures stay honestly unresolved."""
    text = str(exc or "").lower()
    if "no endpoints found" not in text:
        return False
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    return status == 404 or "error code: 404" in text or '"code": 404' in text or "'code': 404" in text


def _is_tos_rejection(exc: BaseException) -> bool:
    """True for an OpenRouter ToS-policy 403 that never reached generation.

    OpenRouter rejects the request with HTTP 403 and the body signature
    "prohibited due to a violation of provider Terms Of Service" (the OpenAI SDK
    raises it as PermissionDeniedError with status_code=403; the match here is
    structural-or-textual so a transport wrapper that stringifies the error
    still settles) BEFORE any upstream generation — the audited CLB incident showed 0 llm_usage events
    and 0 billed tokens across 240 such rejections while the ledger accumulated
    a ~$479 phantom unresolved bound. Settling the attempt at a confirmed $0
    releases the conservative reservation instead of holding that phantom bound
    against the budget fence. Deliberately narrow, mirroring
    _is_pre_routing_rejection: BOTH the 403 status and the exact ToS body
    signature are required (and the caller additionally gates on
    provider == "openrouter"), so generic 401/403/quota failures — where the
    outcome is genuinely unknown — stay honestly unresolved."""
    text = str(exc or "").lower()
    if "prohibited due to a violation of provider terms of service" not in text:
        return False
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    return status == 403 or "error code: 403" in text or '"code": 403' in text or "'code': 403" in text


def _terminalize_failed_attempt(reservation: AttemptReservation, exc: BaseException) -> str:
    """Route a raised provider send to its honest terminal ledger state."""
    provider = str(reservation.provider or "").strip().lower()
    if provider == "openrouter" and _is_pre_routing_rejection(exc):
        _transition(
            reservation,
            "settled",
            cost_usd=0.0,
            cost_final=True,
            settle_reason="pre_routing_rejection",
        )
        return "settled"
    elif provider == "openrouter" and _is_tos_rejection(exc):
        _transition(
            reservation,
            "settled",
            cost_usd=0.0,
            cost_final=True,
            settle_reason="tos_rejection",
        )
        return "settled"
    else:
        mark_unresolved(reservation, f"{type(exc).__name__}: {exc}")
        return "unresolved"


def _provider_exception_facts(exc: BaseException) -> Tuple[bool, Optional[int], str]:
    """Extract response facts using the classifier's typed HTTP aliases."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    status = getattr(exc, "status", None) if status is None else status
    status = getattr(exc, "code", None) if status is None else status
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None

    body = getattr(exc, "body", None)
    if body is None and response is not None and callable(getattr(response, "json", None)):
        try:
            body = response.json()
        except Exception:
            body = None
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else body
    code = ""
    if isinstance(error, dict):
        code = str(error.get("code") or error.get("type") or "").strip()
    provider_response_observed = bool(status is not None or response is not None or body is not None)
    return provider_response_observed, status, code


def _attach_attempt_capture(reservation: AttemptReservation, state: str, exc: BaseException) -> PhysicalAttemptCapture:
    observed, status, code = _provider_exception_facts(exc)
    capture = PhysicalAttemptCapture(
        attempt_id=reservation.attempt_id,
        model=reservation.model,
        provider=reservation.provider,
        state=state,
        provider_response_observed=observed,
        provider_status_code=status,
        provider_code=code,
        logical_call_attempt_ids=tuple(_ATTEMPT_COLLECTOR.get() or (reservation.attempt_id,)),
    )
    try:
        setattr(exc, "physical_attempt_capture", capture)
    except Exception:
        log.debug("Provider exception does not accept attempt capture", exc_info=True)
    return capture


def execute_physical_attempt(
    request: AttemptRequest,
    send: Callable[[], Any],
    *,
    extractor: Callable[[Any], Tuple[Dict[str, Any], Optional[float], bool]] = usage_from_response,
) -> Any:
    """Execute one synchronous provider send with durable lifecycle accounting."""
    reservation = reserve_attempt(request)
    mark_dispatched(reservation)
    try:
        response = send()
    except BaseException as exc:
        terminal_state = "dispatched"
        try:
            terminal_state = _terminalize_failed_attempt(reservation, exc)
        except Exception:
            log.exception("Failed to mark provider attempt unresolved: %s", reservation.attempt_id)
        _attach_attempt_capture(reservation, terminal_state, exc)
        raise
    try:
        usage, cost, final = extractor(response)
        settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
        _observe_token_density(request, usage)
    except Exception as exc:
        # The provider response may already be paid and useful.  Preserve it;
        # extractor and persistence failures both leave an unresolved upper bound.
        log.exception("Failed to account paid provider response: %s", reservation.attempt_id)
        try:
            mark_unresolved(reservation, f"post_response_accounting_failed:{type(exc).__name__}")
        except Exception:
            log.exception("Failed to mark post-response accounting failure unresolved")
    return response


async def execute_physical_attempt_async(
    request: AttemptRequest,
    send: Callable[[], Any],
    *,
    extractor: Callable[[Any], Tuple[Dict[str, Any], Optional[float], bool]] = usage_from_response,
) -> Any:
    reservation = reserve_attempt(request)
    mark_dispatched(reservation)
    try:
        response = await send()
    except BaseException as exc:
        terminal_state = "dispatched"
        try:
            terminal_state = _terminalize_failed_attempt(reservation, exc)
        except Exception:
            log.exception("Failed to mark provider attempt unresolved: %s", reservation.attempt_id)
        _attach_attempt_capture(reservation, terminal_state, exc)
        raise
    try:
        usage, cost, final = extractor(response)
        settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
        _observe_token_density(request, usage)
    except Exception as exc:
        log.exception("Failed to account paid provider response: %s", reservation.attempt_id)
        try:
            mark_unresolved(reservation, f"post_response_accounting_failed:{type(exc).__name__}")
        except Exception:
            log.exception("Failed to mark post-response accounting failure unresolved")
    return response


def _legacy_snapshot(root: pathlib.Path) -> Tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    events_path = root / "logs" / "events.jsonl"
    state_path = root / "state" / "state.json"
    settings_path = pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH") or root / "settings.json")
    sources = {"events.jsonl": events_path, "state.json": state_path}
    snapshots: Dict[str, bytes] = {}
    for name, path in sources.items():
        try:
            snapshots[name] = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UsageAccountingError(f"cannot snapshot legacy usage source {path}: {exc}") from exc
    hashes = {name: hashlib.sha256(snapshots[name]).hexdigest() if name in snapshots else "" for name in sources}
    # Settings are owner-secret state: prove non-mutation by hash, but never copy
    # their contents into the usage archive.
    try:
        hashes["settings.json"] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        hashes["settings.json"] = ""
    except OSError as exc:
        raise UsageAccountingError(f"cannot hash settings file {settings_path}: {exc}") from exc
    rows: list[Dict[str, Any]] = []
    try:
        event_text = snapshots.get("events.jsonl", b"").decode("utf-8")
        for line_no, line in enumerate(event_text.splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("type") == "llm_usage":
                rows.append({**value, "_legacy_line": line_no})
    except UnicodeDecodeError:
        pass
    try:
        state = json.loads(snapshots.get("state.json", b"{}").decode("utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        state = {}

    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    archive = root / "archive" / "usage_import" / combined
    archive.mkdir(parents=True, exist_ok=True)
    for name, payload in snapshots.items():
        target = archive / name
        if target.exists():
            if target.read_bytes() != payload:
                raise UsageAccountingError(f"legacy usage archive mismatch: {target}")
        else:
            _write_bytes_atomic_fsync(target, payload)
            try:
                target.chmod(0o400)
            except OSError:
                pass
    atomic_write_json(archive / "sha256.json", hashes, trailing_newline=True, fsync=True)
    return rows, state, hashes


def ensure_legacy_imported(
    drive_root: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """One resumable import of legacy usage telemetry and the state cost delta."""
    root = _drive_root(drive_root)
    completed = _completed_import_watermark(root)
    if completed is not None:
        return completed
    # Separate from the hot budget lock: source snapshot/archive may do I/O,
    # while concurrent startup importers still serialize on one generation.
    with _named_lock(root, "usage_import.lock", timeout_sec=60.0, stale_sec=600.0):
        return _ensure_legacy_imported_locked(root)


def _completed_import_watermark(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads((root / IMPORT_REL).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("completed") else None


def _ensure_legacy_imported_locked(
    root: pathlib.Path,
) -> Dict[str, Any]:
    watermark = root / IMPORT_REL
    existing = _completed_import_watermark(root)
    if existing is not None:
        return existing

    legacy_rows, state, hashes = _legacy_snapshot(root)
    baseline_source = "state.json"
    candidates: list[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    imported_cost = 0.0
    usage_count = 0
    for event in legacy_rows:
        line_no = int(event.pop("_legacy_line", 0) or 0)
        legacy_usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        fingerprint = hashlib.sha256(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        task_id = str(event.get("task_id") or "")
        root_task_id = str(event.get("root_task_id") or task_id)
        raw_cost = event.get("cost")
        if raw_cost is None:
            raw_cost = legacy_usage.get("cost", legacy_usage.get("total_cost"))
        cost = _number(raw_cost)

        def legacy_int(field: str, *aliases: str) -> int:
            for candidate in (field, *aliases):
                value = event.get(candidate)
                if value in (None, ""):
                    value = legacy_usage.get(candidate)
                try:
                    return max(0, int(float(value or 0)))
                except (TypeError, ValueError):
                    continue
            return 0

        prompt = legacy_int("prompt_tokens", "input_tokens")
        completion = legacy_int("completion_tokens", "output_tokens")
        provider = str(event.get("provider") or event.get("api_key_type") or "unknown")
        if cost == 0 and (prompt or completion) and provider != "local":
            cost = None  # legacy zero may mean unknown pricing, never "free"
        usage_count += 1
        if cost is not None:
            imported_cost += cost
        candidates.append(
            {
                "kind": "legacy_usage",
                "attempt_id": f"legacy-{fingerprint[:24]}",
                "state": "settled",
                "model": str(event.get("model") or ""),
                "provider": provider,
                "cost_usd": cost,
                "cost_final": bool(cost is not None and not event.get("cost_estimated")),
                "reservation_upper_bound_usd": None,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_tokens": legacy_int("cached_tokens", "cache_read_input_tokens"),
                "cache_write_tokens": legacy_int("cache_write_tokens", "cache_creation_input_tokens"),
                "prompt_cache_ttl": str(
                    event.get("prompt_cache_ttl") or legacy_usage.get("prompt_cache_ttl") or ""
                ),
                "task_id": task_id,
                "root_task_id": root_task_id,
                "parent_task_id": str(event.get("parent_task_id") or ""),
                "category": str(event.get("category") or "legacy"),
                "source": "legacy_llm_usage",
                "legacy_line": line_no,
            }
        )
    legacy_calls = max(0, int(state.get("spent_calls") or state.get("calls") or 0))
    metadata_count = max(0, legacy_calls - usage_count)
    if metadata_count:
        identity = hashlib.sha256(
            f"legacy-metadata:{metadata_count}:{hashes.get('state.json', '')}".encode()
        ).hexdigest()
        candidates.append(
            {
                "kind": "legacy_metadata",
                "attempt_id": f"legacy-{identity[:24]}",
                "state": "unresolved",
                "model": "",
                "provider": "legacy",
                "reservation_upper_bound_usd": None,
                "ambiguous_call_count": metadata_count,
                "task_id": "",
                "root_task_id": "",
                "parent_task_id": "",
                "category": "legacy",
                "source": "legacy_state_call_delta",
            }
        )
    state_spent = _number(state.get("spent_usd")) or 0.0
    delta = round(max(0.0, state_spent - imported_cost), 6)
    if delta:
        identity = hashlib.sha256(f"legacy-delta:{delta:.6f}:{hashes.get('state.json', '')}".encode()).hexdigest()
        candidates.append(
            {
                "kind": "legacy_delta",
                "attempt_id": f"legacy-{identity[:24]}",
                "state": "settled",
                "model": "",
                "provider": "legacy",
                "cost_usd": delta,
                "cost_final": False,
                "reservation_upper_bound_usd": None,
                "task_id": "",
                "root_task_id": "",
                "parent_task_id": "",
                "category": "legacy",
                "source": "legacy_state_delta",
            }
        )

    with _locked(root):
        current_watermark = _completed_import_watermark(root)
        if current_watermark is not None:
            return current_watermark
        records = _read_records_locked(root)
        existing_ids = {str(row.get("attempt_id") or "") for row in records}
        missing = [row for row in candidates if row["attempt_id"] not in existing_ids]
        _append_rows_locked(root, records, missing)
        result = {
            "completed": True,
            "completed_at": utc_now_iso(),
            "source_sha256": hashes,
            "legacy_baseline_source": baseline_source,
            "legacy_baseline_spent_usd": state_spent,
            "legacy_baseline_spent_calls": legacy_calls,
            "legacy_usage_count": usage_count,
            "legacy_metadata_count": metadata_count,
            "legacy_delta_usd": delta,
            # The legacy schema has no trustworthy typed test/operator bit.
            # Never invent exclusions from names, task ids, or source strings.
            "quarantined_test_operator_rows": 0,
            "test_operator_quarantine_policy": "typed_evidence_only_no_inference",
            "events_exceed_state_calls": max(0, usage_count - legacy_calls),
            "events_exceed_state_usd": round(max(0.0, imported_cost - state_spent), 6),
            "rows_appended": len(missing),
        }
        atomic_write_json(watermark, result, trailing_newline=True, fsync=True)
    append_jsonl(root / "logs" / "events.jsonl", {"type": "usage_import_completed", **result})
    return result
