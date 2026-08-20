"""Durable physical-model-attempt accounting.

The append-only JSONL ledger is the monetary authority.  Existing
``llm_usage`` events and ``state.json`` remain compatibility projections and
carry ledger attempt ids, so they can never become a second charge source.

The implementation is deliberately small: no hash chain, fanout reservation,
epoch/reconcile platform, or per-attempt snapshot database.  A projection is
replayed from validated records under the same short cross-process lock used
for budget check + append + fsync; network I/O always happens outside that lock.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Sequence, Tuple

from ouroboros.pricing import estimate_cost_optional
from ouroboros._usage_response import usage_from_response
# One-way seam: usage_ledger owns bytes/validation; this module owns policy.
from ouroboros.usage_ledger import (  # noqa: F401 — re-exported substrate
    LEDGER_REL,
    QUARANTINE_REL,
    LedgerResumeState,
    UsageAccountingError,
    UsageLedgerCorrupt,
    _append_bytes_fsync,
    _append_rows_locked,
    _drive_root,
    _final_rows,
    _ledger_resume_state,
    _locked,
    _named_lock,
    _number,
    _read_new_records_locked,
    _read_records_locked,
    _TERMINAL,
    _validate_records,
    _write_bytes_atomic_fsync,
)
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso  # noqa: F401 -- the accounting module keeps its historical import surface for the L-C2 leaf
from ouroboros._usage_rows import (  # noqa: F401  (re-exported substrate vocabulary)
    _breakdown_bucket,
    _physical_call_count,
    _summary,
    _with_integrity,
    _with_limit,
)

log = logging.getLogger(__name__)
__all__ = (
    "AttemptRequest", "AttemptReservation", "BudgetExceeded", "PhysicalAttemptCapture",
    "PhysicalAttemptContext", "PhysicalAttemptLimitExceeded", "PhysicalAttemptPreconditionFailed",
    "PhysicalAttemptPreparationFailed",
    "UsageAccountingError", "UsageLedgerCorrupt", "UsageScope", "capture_attempt_ids",
    "bind_physical_attempt_context", "current_physical_attempt_context",
    "current_physical_attempt_predicate", "current_usage_scope",
    "ensure_legacy_imported", "execute_physical_attempt", "execute_physical_attempt_async",
    "last_physical_attempt_capture", "last_root_accounting", "physical_attempt_capture_from_exception",
    "mark_dispatched", "mark_unresolved", "physical_attempt_limit",
    "record_subscription_session",
    "record_unmetered_external_dispatch", "refresh_root_accounting",
    "release_attempt", "reserve_attempt", "settle_attempt",
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
_PHYSICAL_CONTEXT: contextvars.ContextVar[Optional["PhysicalAttemptContext"]] = contextvars.ContextVar(
    "ouroboros_physical_attempt_context", default=None
)
_PHYSICAL_PREDICATE: contextvars.ContextVar[Optional[Callable[["AttemptRequest"], Any]]] = contextvars.ContextVar(
    "ouroboros_physical_attempt_predicate", default=None
)
_LAST_PHYSICAL_ATTEMPT: contextvars.ContextVar[Optional["PhysicalAttemptCapture"]] = contextvars.ContextVar(
    "ouroboros_last_physical_attempt", default=None
)

_ROOT_ACCOUNTING_TELEMETRY: Dict[str, Dict[str, Any]] = {}
_ROOT_ACCOUNTING_TELEMETRY_LOCK = threading.Lock()
_ROOT_ACCOUNTING_TELEMETRY_CAP = 64


def _stash_root_accounting(
    root_task_id: str,
    accounted_usd: Optional[float],
    root_limit_usd: Optional[float],
) -> None:
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return
    with _ROOT_ACCOUNTING_TELEMETRY_LOCK:
        if (
            root_task_id not in _ROOT_ACCOUNTING_TELEMETRY
            and len(_ROOT_ACCOUNTING_TELEMETRY) >= _ROOT_ACCOUNTING_TELEMETRY_CAP
        ):
            oldest = min(
                _ROOT_ACCOUNTING_TELEMETRY,
                key=lambda key: _ROOT_ACCOUNTING_TELEMETRY[key]["updated_monotonic"],
            )
            _ROOT_ACCOUNTING_TELEMETRY.pop(oldest, None)
        _ROOT_ACCOUNTING_TELEMETRY[root_task_id] = {
            "accounted_usd": None if accounted_usd is None else float(accounted_usd),
            "root_limit_usd": None if root_limit_usd is None else float(root_limit_usd),
            "updated_monotonic": time.monotonic(),
        }


def last_root_accounting(root_task_id: str) -> Optional[Dict[str, Any]]:
    """Newest process-local root snapshot, including in-flight holds."""
    with _ROOT_ACCOUNTING_TELEMETRY_LOCK:
        entry = _ROOT_ACCOUNTING_TELEMETRY.get(str(root_task_id or "").strip())
        if entry is None:
            return None
        entry = dict(entry)
    entry["age_sec"] = max(0.0, time.monotonic() - entry.pop("updated_monotonic"))
    return entry


def refresh_root_accounting(
    drive_root: pathlib.Path | str | None,
    root_task_id: str,
    *,
    max_age_sec: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Refresh a stale root snapshot; on failure return stale/None, never fake $0."""
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return None
    cached = last_root_accounting(root_task_id)
    if cached is not None and max_age_sec > 0 and cached["age_sec"] <= max_age_sec:
        return cached
    try:
        projection = usage_projection(drive_root, root_task_id=root_task_id)
        _stash_root_accounting(
            root_task_id,
            _number(projection.get("accounted_usd")),
            _number(projection.get("limit_usd")),
        )
        return last_root_accounting(root_task_id)
    except Exception:
        log.debug("root accounting refresh failed for %s", root_task_id, exc_info=True)
        return cached


class BudgetExceeded(UsageAccountingError):
    """Raised before dispatch when a known budget would be exceeded."""

    def __init__(self, message: str, *, limit_scope: str = "global", root_task_id: str = "") -> None:
        super().__init__(message)
        self.limit_scope = str(limit_scope or "global")
        self.root_task_id = str(root_task_id or "")


class PhysicalAttemptLimitExceeded(UsageAccountingError):
    """Raised before a provider send would exceed the caller's actor-local rail."""


class PhysicalAttemptPreparationFailed(UsageAccountingError):
    """An inspectable candidate could not be persisted before dispatch."""

    def __init__(self, message: str, *, attempt_id: str = "") -> None:
        super().__init__(message)
        self.attempt_id = str(attempt_id or "")


class PhysicalAttemptPreconditionFailed(PhysicalAttemptPreparationFailed):
    """The host rejected an immutable final-candidate fact before dispatch."""
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
class PhysicalAttemptContext:
    profile: Literal["owner_max", "owner_low", "task_local_low"]
    rendered_mode: Literal["max", "low"]
    measurement_basis: Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]
    route_fp: str
    round_id: str
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    context_target_miss: bool
    automatic_pass_used: bool


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
    # Applied payload TTL; empty construction sites fall back to the owner SSOT.
    prompt_cache_ttl: str = ""
    candidate_raw_sha256: Optional[str] = None
    candidate_raw_size_bytes: Optional[int] = None
    candidate_context_sha256: Optional[str] = None
    candidate_context_size_bytes: Optional[int] = None
    candidate_measurement_kind: Literal["canonical_json_v1", "opaque"] = "opaque"
    physical_context: Optional[PhysicalAttemptContext] = None


@dataclass(frozen=True)
class AttemptReservation:
    attempt_id: str
    drive_root: pathlib.Path
    model: str
    provider: str
    reservation_upper_bound_usd: Optional[float]


@dataclass(frozen=True)
class PhysicalAttemptCapture:
    attempt_id: str
    model: str
    provider: str
    state: Literal["reserved", "released", "dispatched", "settled", "unresolved"]
    candidate_measurement_kind: Literal["canonical_json_v1", "opaque"]
    max_completion_tokens: int = 0
    candidate_raw_sha256: Optional[str] = None
    candidate_raw_size_bytes: Optional[int] = None
    candidate_context_sha256: Optional[str] = None
    candidate_context_size_bytes: Optional[int] = None
    candidate_manifest_ref: Optional[Dict[str, Any]] = None
    physical_context: Optional[PhysicalAttemptContext] = None
    provider_status_code: Optional[int] = None
    provider_code: str = ""
    provider_error_type: str = ""
    provider_error: str = ""


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
def bind_physical_attempt_context(
    context: PhysicalAttemptContext,
    candidate_predicate: Optional[Callable[[AttemptRequest], Any]] = None,
) -> Iterator[PhysicalAttemptContext]:
    """Bind frozen Main metadata and an optional final-fact predicate."""
    if not isinstance(context, PhysicalAttemptContext):
        raise TypeError("physical attempt context must be PhysicalAttemptContext")
    context_token = _PHYSICAL_CONTEXT.set(context)
    predicate_token = _PHYSICAL_PREDICATE.set(candidate_predicate)
    try:
        yield context
    finally:
        _PHYSICAL_PREDICATE.reset(predicate_token)
        _PHYSICAL_CONTEXT.reset(context_token)


def current_physical_attempt_context() -> Optional[PhysicalAttemptContext]:
    return _PHYSICAL_CONTEXT.get()


def current_physical_attempt_predicate() -> Optional[Callable[[AttemptRequest], Any]]:
    return _PHYSICAL_PREDICATE.get()


def last_physical_attempt_capture() -> Optional[PhysicalAttemptCapture]:
    return _LAST_PHYSICAL_ATTEMPT.get()


def physical_attempt_capture_from_exception(exc: BaseException) -> Optional[PhysicalAttemptCapture]:
    capture = getattr(exc, "physical_attempt_capture", None)
    return capture if isinstance(capture, PhysicalAttemptCapture) else last_physical_attempt_capture()


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


from ouroboros._usage_rows_memo import (  # noqa: F401,E402  (re-exported seam)
    _LedgerRowsMemo,
    _ROWS_MEMO,
    _ROWS_MEMO_LOCK,
    _memoized_final_rows,
    _render_cached,
)


def usage_projection(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    global_limit_usd: Optional[float] = None,
    include_roots: bool = True,
) -> Dict[str, Any]:
    """Return a replayed global projection, or one root/subtree projection.

    ``include_roots=False`` skips building the per-root ``by_root`` map for
    hot-path readers that never consume it (``/api/state``); the slim result
    still carries ``limit_usd``/``remaining_known_usd`` — the two fields
    ``budget_remaining`` consumes. The default keeps the full contract."""
    root = _drive_root(drive_root)
    if root_task_id:
        cache_key = ("usage_projection", root_task_id, "", None, True)

        def render_root(final: list, integrity_degraded: bool) -> Dict[str, Any]:
            rows = [row for row in final if str(row.get("root_task_id") or "") == root_task_id]
            limits = [_number(row.get("root_limit_usd")) for row in rows]
            known_limits = [value for value in limits if value is not None]
            result = _with_limit(_summary(rows), min(known_limits) if known_limits else None)
            return _with_integrity(result, integrity_degraded)

        return _render_cached(root, cache_key, render_root)
    if global_limit_usd is not None:
        configured_limit = max(0.0, float(global_limit_usd))
    else:
        try:
            configured_limit = float(os.environ.get("TOTAL_BUDGET", "200") or 0.0)
        except (TypeError, ValueError):
            configured_limit = 200.0
    apply_limit = global_limit_usd is not None or configured_limit > 0
    cache_key = (
        "usage_projection", "", "",
        configured_limit if apply_limit else None,
        include_roots,
    )

    def render_global(final: list, integrity_degraded: bool) -> Dict[str, Any]:
        result = (
            _with_limit(_summary(final), configured_limit) if apply_limit else _summary(final)
        )
        if include_roots:
            grouped_rows: Dict[str, list] = {}
            for row in final:
                rid = str(row.get("root_task_id") or "")
                if rid:
                    grouped_rows.setdefault(rid, []).append(row)
            result["by_root"] = {}
            for rid in sorted(grouped_rows):
                root_rows = grouped_rows[rid]
                known_limits = [
                    value
                    for value in (_number(row.get("root_limit_usd")) for row in root_rows)
                    if value is not None
                ]
                result["by_root"][rid] = _with_integrity(
                    _with_limit(_summary(root_rows), min(known_limits) if known_limits else None),
                    integrity_degraded,
                )
        return _with_integrity(result, integrity_degraded)

    return _render_cached(root, cache_key, render_global)


def usage_breakdown(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    task_id: str = "",
) -> Dict[str, Any]:
    """Read-only physical-call/token/cost buckets from validated ledger finals."""
    root = _drive_root(drive_root)
    cache_key = ("usage_breakdown", root_task_id, task_id, None, True)

    def render(final: list, integrity_degraded: bool) -> Dict[str, Any]:
        rows = final
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
            # Execution-axis filter (v6.91): the delegated (subscription-harness) rows
            # only — a VIEW over the same rows for "where did the money go" readers,
            # never a third monetary sum or authority. Disclosed-free sessions settle
            # at $0 here; undisclosed spend stays in `unknown`.
            "delegated": _with_integrity(
                _breakdown_bucket([row for row in rows if str(row.get("kind") or "") == "subscription_session"]),
                integrity_degraded,
            ),
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

    return _render_cached(root, cache_key, render)


def _reservation_cost(request: AttemptRequest) -> Optional[float]:
    explicit = request.max_budget_usd if request.max_budget_usd is not None else request.reservation_usd
    if explicit is not None:
        return _number(explicit)
    if request.force_unknown_reservation:
        return None
    if str(request.provider or "").lower() == "local":
        return 0.0
    prompt_tokens = max(0, int(request.prompt_tokens_estimate or 0))
    # OpenAI-family chars/4 estimates keep the measured 1.10 reservation envelope.
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
    prompt_cache_ttl: Optional[str] = None
    if cache_write_tokens:
        # Price the applied candidate TTL; unknown construction sites use the owner SSOT.
        from ouroboros.config import PROMPT_CACHE_TTL_SCALE, resolve_prompt_cache_ttl

        prompt_cache_ttl = str(request.prompt_cache_ttl or "").strip().lower()
        if prompt_cache_ttl not in PROMPT_CACHE_TTL_SCALE:
            # An inspectable marker-free candidate writes no cache at all. Keep
            # the historical conservative base-tier reservation without
            # misreporting a TTL on the eventual physical settlement. Opaque
            # construction sites still fall back to the owner setting.
            prompt_cache_ttl = (
                "default"
                if request.candidate_measurement_kind == "canonical_json_v1"
                else resolve_prompt_cache_ttl()
            )
    return estimate_cost_optional(
        request.model,
        prompt_tokens,
        max(0, int(request.max_completion_tokens or 0)),
        cache_usage={"cache_write_tokens": cache_write_tokens,
                     "prompt_cache_ttl": prompt_cache_ttl},
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
    """Read-only all-slot admission using the normal reservation math; fail open."""
    result: Dict[str, Any] = {
        "fits": True,
        "estimated_wave_usd": None,
        "remaining_usd": None,
        "limit_usd": None,
        "slots": len(list(models or [])),
        "unpriced_slots": 0,
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
                # Unknown contributes no invented price and remains explicitly counted.
                result["unpriced_slots"] = int(result.get("unpriced_slots") or 0) + 1
                continue
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
        configured = float(os.environ.get("TOTAL_BUDGET", "200") or 0.0)
        return configured if configured > 0 else float("inf")
    except (TypeError, ValueError):
        return 200.0


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


_CANDIDATE_ROW_FIELDS = (
    "candidate_raw_sha256", "candidate_raw_size_bytes", "candidate_context_sha256",
    "candidate_context_size_bytes", "candidate_measurement_kind", "physical_context",
    "candidate_manifest_ref",
)


def _candidate_request_fields(request: AttemptRequest) -> Dict[str, Any]:
    return {
        "candidate_raw_sha256": request.candidate_raw_sha256,
        "candidate_raw_size_bytes": request.candidate_raw_size_bytes,
        "candidate_context_sha256": request.candidate_context_sha256,
        "candidate_context_size_bytes": request.candidate_context_size_bytes,
        "candidate_measurement_kind": request.candidate_measurement_kind,
        "physical_context": asdict(request.physical_context) if request.physical_context else None,
    }


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
        root_rows: Optional[list[Dict[str, Any]]] = None
        root_limit: Optional[float] = None
        if scope.root_task_id and scope.root_limit_usd is not None:
            root_rows = [row for row in finals if str(row.get("root_task_id") or "") == scope.root_task_id]
            root_accounted = float(_summary(root_rows)["accounted_usd"])
            root_limit = max(0.0, float(scope.root_limit_usd))
            # Piggyback the measured pre-append subtree sum on this locked read.
            _stash_root_accounting(scope.root_task_id, root_accounted, root_limit)
            if root_limit <= 0 or root_accounted >= root_limit - 1e-9 or (
                bound is not None and root_accounted + bound > root_limit + 1e-9
            ):
                raise BudgetExceeded(
                    f"root model budget exhausted for {scope.root_task_id}: "
                    f"accounted=${root_accounted:.6f}, limit=${root_limit:.6f}",
                    limit_scope="root",
                    root_task_id=scope.root_task_id,
                )
        appended = _append_rows_locked(
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
                    **_candidate_request_fields(request),
                }
            ],
        )
        if root_rows is not None:
            _stash_root_accounting(
                scope.root_task_id,
                float(_summary([*root_rows, *appended])["accounted_usd"]),
                root_limit,
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
    return _append_single_settled_row(root, row, comparable=(
        "kind", "model", "provider", "task_id", "root_task_id", "parent_task_id",
        "category", "source", "prompt_tokens", "completion_tokens",
        "external_dispatch_id_sha256",
    ))


def _append_single_settled_row(
    root: pathlib.Path, row: Dict[str, Any], *, comparable: Sequence[str],
) -> str:
    """Idempotently append a one-shot settled row; a replay under a DIFFERENT identity
    is a conflict, never a silent overwrite. Shared by every single-row kind."""
    attempt_id = str(row["attempt_id"])
    with _locked(root):
        records = _read_records_locked(root)
        existing = _final_rows(records).get(attempt_id)
        if existing is not None:
            if any(existing.get(key) != row.get(key) for key in comparable):
                raise UsageAccountingError(f"conflicting settled-row identity: {attempt_id}")
            return attempt_id
        _append_rows_locked(root, records, [row])
    return attempt_id


def record_subscription_session(
    session_id: str,
    *,
    drive_root: pathlib.Path | str | None = None,
    route: str,
    model: str = "",
    task_id: str = "",
    root_task_id: str = "",
    parent_task_id: str = "",
    category: str = "subagent",
    source: str = "delegated_subagent",
    prompt_tokens: int | None = 0,
    completion_tokens: int | None = 0,
    cached_tokens: int | None = None,
    reset_at: str = "",
    spend_usd: float | None = None,
    spend_estimated: bool = False,
    credential_profile_id: str = "",
    access_profile: str = "",
) -> str:
    """Record one idempotent subscription session; None remains undisclosed."""
    stable_id, route_id = str(session_id or "").strip(), str(route or "").strip()
    if not stable_id or not route_id:
        raise UsageAccountingError("subscription session requires a stable session_id and route")
    bound = _CURRENT_SCOPE.get() or UsageScope()
    root = _drive_root(drive_root or bound.drive_root)
    ensure_legacy_imported(root)
    identity = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    attempt_id = f"session-{identity[:24]}"
    row = {
        "kind": "subscription_session",
        "attempt_id": attempt_id,
        "state": "settled",
        "model": str(model or ""),
        "provider": route_id,
        # None is undisclosed; zero is genuinely free; estimated amounts are non-final.
        "cost_usd": None if spend_usd is None else round(float(spend_usd), 6),
        "cost_final": spend_usd is not None and not spend_estimated,
        "reservation_upper_bound_usd": None if spend_usd is None else round(float(spend_usd), 6),
        "pricing_known": spend_usd is not None,
        "prompt_tokens": None if prompt_tokens is None else max(0, int(prompt_tokens)),
        "completion_tokens": None if completion_tokens is None else max(0, int(completion_tokens)),
        # Cached tokens are a separate axis because harness semantics differ.
        "cached_tokens": None if cached_tokens is None else max(0, int(cached_tokens)),
        "task_id": str(task_id or bound.task_id or ""),
        "root_task_id": str(root_task_id or bound.root_task_id or task_id or bound.task_id or ""),
        "parent_task_id": str(parent_task_id or bound.parent_task_id or ""),
        "category": str(category or bound.category or "subagent"),
        "source": str(source or bound.source or "delegated_subagent"),
        "subscription_route": route_id,
        "subscription_reset_at": str(reset_at or ""),
        # Empty profile/access means the engine reported none.
        "credential_profile_id": str(credential_profile_id or ""),
        "access_profile": str(access_profile or ""),
        "session_id_sha256": identity,
    }
    return _append_single_settled_row(root, row, comparable=(
        "kind", "model", "provider", "task_id", "root_task_id", "parent_task_id",
        "category", "source", "subscription_route", "session_id_sha256",
    ))


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
            **{key: current.get(key) for key in _CANDIDATE_ROW_FIELDS if key in current},
            **fields,
        }
        appended = _append_rows_locked(reservation.drive_root, records, [row])
        root_task_id = str(current.get("root_task_id") or "")
        root_limit = _number(current.get("root_limit_usd"))
        if root_task_id and root_limit is not None:
            # Refresh from post-transition finals without another ledger read.
            subtree = [
                r for r in _final_rows([*records, *appended]).values()
                if str(r.get("root_task_id") or "") == root_task_id
            ]
            _stash_root_accounting(
                root_task_id, float(_summary(subtree)["accounted_usd"]), root_limit,
            )
        return appended[0]


def mark_dispatched(
    reservation: AttemptReservation,
    *,
    candidate_manifest_ref: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _claim_physical_dispatch()
    except PhysicalAttemptLimitExceeded:
        release_attempt(
            reservation,
            "physical_attempt_limit",
            candidate_manifest_ref=candidate_manifest_ref,
        )
        raise
    fields = {"candidate_manifest_ref": candidate_manifest_ref} if candidate_manifest_ref else {}
    _transition(reservation, "dispatched", **fields)


def release_attempt(
    reservation: AttemptReservation, reason: str = "not_dispatched", *, candidate_manifest_ref=None,
) -> None:
    _transition(reservation, "released", reason=str(reason or "not_dispatched"), **(
        {"candidate_manifest_ref": candidate_manifest_ref} if candidate_manifest_ref else {}))


def mark_unresolved(reservation: AttemptReservation, reason: str) -> None:
    try:
        from ouroboros.observability import redact_projection

        safe_reason = str(redact_projection(
            str(reason or "provider_outcome_unknown"),
        ).value)
    except Exception:
        safe_reason = "provider_outcome_unknown:redaction_failed"
    _transition(reservation, "unresolved", reason=safe_reason[:500])


def terminalize_abandoned_attempt(
    reservation: AttemptReservation,
    *,
    reason: str,
    usage: Optional[Dict[str, Any]] = None,
) -> str:
    """Close a dead owner attempt from measured usage, else unresolved/released."""
    with _locked(reservation.drive_root):
        current = _final_rows(_read_records_locked(reservation.drive_root)).get(
            reservation.attempt_id
        )
    if current is None:
        return "unknown"
    state = str(current.get("state") or "")
    if state in _TERMINAL:
        return state
    if state == "reserved":
        release_attempt(reservation, reason)
        return "released"
    normalized = dict(usage or {})
    measured = int(
        _number(normalized.get("prompt_tokens") or normalized.get("input_tokens")) or 0
    ) + int(
        _number(normalized.get("completion_tokens") or normalized.get("output_tokens")) or 0
    )
    if measured > 0:
        settle_attempt(reservation, normalized, cost_usd=None, cost_final=False)
        return "settled"
    mark_unresolved(reservation, reason)
    return "unresolved"


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
            cache_usage={"cached_tokens": int(normalized.get("cached_tokens") or 0),
                         "cache_write_tokens": int(normalized.get("cache_write_tokens") or 0),
                         "cache_write_tokens_by_ttl": normalized.get("cache_write_tokens_by_ttl"),
                         "prompt_cache_ttl": str(normalized.get("prompt_cache_ttl") or "")},
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


# Cache-inclusive prompt totals are measurable; GigaChat's semantics remain unknown.
_CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS = frozenset({
    "openrouter", "openai", "openai-compatible", "cloudru", "local", "anthropic",
})


def _observe_token_density(request: AttemptRequest, usage: Optional[Dict[str, Any]]) -> None:
    """Learn density after settlement; unknown cache semantics produce no witness."""
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
            route_fp=str(request.physical_context.route_fp if request.physical_context else ""),
        )
    except Exception:
        log.debug("token-density observation skipped", exc_info=True)


def _is_pre_routing_rejection(exc: BaseException) -> bool:
    """True for the exact free OpenRouter router-side 404 signature."""
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
    """True for the exact free OpenRouter ToS-policy 403 signature."""
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


def _provider_exception_facts(exc: BaseException) -> Tuple[Optional[int], str, str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    payload = getattr(exc, "body", None)
    if payload is None and response is not None and callable(getattr(response, "json", None)):
        try:
            payload = response.json()
        except Exception:
            payload = None
    error = payload.get("error") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else payload
    code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    message = str(exc or "")
    if isinstance(error, dict):
        code = error.get("code", code)
        error_type = error.get("type", error_type)
        details = json.dumps(error, ensure_ascii=False, sort_keys=True, default=str)
        message = f"{message}; provider_error={details}" if message else details
    try:
        from ouroboros.observability import redact_projection
        message = str(redact_projection(message).value)
    except Exception:
        message = f"{type(exc).__name__}: provider error details unavailable"
    return status, str(code or ""), str(error_type or type(exc).__name__), message


def _record_attempt_capture(
    reservation: AttemptReservation,
    request: AttemptRequest,
    state: str,
    *,
    candidate_manifest_ref: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> PhysicalAttemptCapture:
    status, code, error_type, error = _provider_exception_facts(exc) if exc is not None else (None, "", "", "")
    capture = PhysicalAttemptCapture(
        attempt_id=reservation.attempt_id,
        model=reservation.model,
        provider=reservation.provider,
        state=state,  # type: ignore[arg-type]
        candidate_measurement_kind=request.candidate_measurement_kind,
        max_completion_tokens=max(0, int(request.max_completion_tokens or 0)),
        candidate_raw_sha256=request.candidate_raw_sha256,
        candidate_raw_size_bytes=request.candidate_raw_size_bytes,
        candidate_context_sha256=request.candidate_context_sha256,
        candidate_context_size_bytes=request.candidate_context_size_bytes,
        candidate_manifest_ref=dict(candidate_manifest_ref) if candidate_manifest_ref else None,
        physical_context=request.physical_context,
        provider_status_code=status,
        provider_code=code,
        provider_error_type=error_type,
        provider_error=error,
    )
    _LAST_PHYSICAL_ATTEMPT.set(capture)
    if exc is not None:
        try:
            setattr(exc, "physical_attempt_capture", capture)
        except Exception:
            pass
    return capture


def _pre_dispatch_failure(
    reservation: AttemptReservation,
    request: AttemptRequest,
    exc: BaseException,
    *,
    candidate_manifest_ref: Optional[Dict[str, Any]] = None,
) -> BaseException:
    manifest_ref = getattr(exc, "candidate_manifest_ref", None) or candidate_manifest_ref
    capture_state = "reserved"
    try:
        release_attempt(reservation, f"before_dispatch_failed:{type(exc).__name__}", candidate_manifest_ref=manifest_ref)
        capture_state = "released"
    except Exception:
        log.exception("Failed to release pre-dispatch attempt: %s", reservation.attempt_id)
    failure = exc if isinstance(exc, PhysicalAttemptPreparationFailed) else PhysicalAttemptPreparationFailed(
        f"physical candidate preparation failed: {type(exc).__name__}: {exc}",
        attempt_id=reservation.attempt_id,
    )
    _record_attempt_capture(reservation, request, capture_state, candidate_manifest_ref=manifest_ref, exc=failure)
    return failure


def execute_physical_attempt(
    request: AttemptRequest,
    send: Callable[[], Any],
    *,
    extractor: Callable[[Any], Tuple[Dict[str, Any], Optional[float], bool]] = usage_from_response,
    before_dispatch: Optional[Callable[[AttemptReservation], Optional[Dict[str, Any]]]] = None,
) -> Any:
    """Execute one synchronous provider send with durable lifecycle accounting."""
    _LAST_PHYSICAL_ATTEMPT.set(None)
    reservation = reserve_attempt(request)
    manifest_ref = None
    try:
        manifest_ref = before_dispatch(reservation) if before_dispatch is not None else None
        if manifest_ref is not None and not isinstance(manifest_ref, dict):
            raise TypeError("before_dispatch must return a manifest ref object or None")
        mark_dispatched(reservation, candidate_manifest_ref=manifest_ref)
        _record_attempt_capture(reservation, request, "dispatched", candidate_manifest_ref=manifest_ref)
    except BaseException as exc:
        if isinstance(exc, PhysicalAttemptLimitExceeded):
            _record_attempt_capture(
                reservation,
                request,
                "released",
                candidate_manifest_ref=manifest_ref,
                exc=exc,
            )
            raise
        failure = _pre_dispatch_failure(
            reservation,
            request,
            exc,
            candidate_manifest_ref=manifest_ref,
        )
        if failure is exc:
            raise
        raise failure from exc
    try:
        response = send()
    except BaseException as exc:
        terminal_state = "dispatched"
        try:
            terminal_state = _terminalize_failed_attempt(reservation, exc)
        except Exception:
            log.exception("Failed to mark provider attempt unresolved: %s", reservation.attempt_id)
        _record_attempt_capture(
            reservation, request, terminal_state, candidate_manifest_ref=manifest_ref, exc=exc,
        )
        raise
    terminal_state = "settled"
    try:
        usage, cost, final = extractor(response)
        usage = dict(usage or {})
        if request.prompt_cache_ttl and not usage.get("prompt_cache_ttl"):
            usage["prompt_cache_ttl"] = request.prompt_cache_ttl
        settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
        _observe_token_density(request, usage)
    except Exception as exc:
        # Preserve a paid/useful response; accounting failure leaves an open bound.
        log.exception("Failed to account paid provider response: %s", reservation.attempt_id)
        terminal_state = "dispatched"
        try:
            mark_unresolved(reservation, f"post_response_accounting_failed:{type(exc).__name__}")
            terminal_state = "unresolved"
        except Exception:
            log.exception("Failed to mark post-response accounting failure unresolved")
    _record_attempt_capture(reservation, request, terminal_state, candidate_manifest_ref=manifest_ref)
    return response


async def execute_physical_attempt_async(
    request: AttemptRequest,
    send: Callable[[], Any],
    *,
    extractor: Callable[[Any], Tuple[Dict[str, Any], Optional[float], bool]] = usage_from_response,
    before_dispatch: Optional[Callable[[AttemptReservation], Any]] = None,
) -> Any:
    _LAST_PHYSICAL_ATTEMPT.set(None)
    reservation = reserve_attempt(request)
    manifest_ref = None
    try:
        if before_dispatch is not None:
            manifest_ref = before_dispatch(reservation)
            if hasattr(manifest_ref, "__await__"):
                manifest_ref = await manifest_ref
        if manifest_ref is not None and not isinstance(manifest_ref, dict):
            raise TypeError("before_dispatch must return a manifest ref object or None")
        mark_dispatched(reservation, candidate_manifest_ref=manifest_ref)
        _record_attempt_capture(reservation, request, "dispatched", candidate_manifest_ref=manifest_ref)
    except BaseException as exc:
        if isinstance(exc, PhysicalAttemptLimitExceeded):
            _record_attempt_capture(
                reservation,
                request,
                "released",
                candidate_manifest_ref=manifest_ref,
                exc=exc,
            )
            raise
        failure = _pre_dispatch_failure(
            reservation,
            request,
            exc,
            candidate_manifest_ref=manifest_ref,
        )
        if failure is exc:
            raise
        raise failure from exc
    try:
        response = await send()
    except BaseException as exc:
        terminal_state = "dispatched"
        try:
            terminal_state = _terminalize_failed_attempt(reservation, exc)
        except Exception:
            log.exception("Failed to mark provider attempt unresolved: %s", reservation.attempt_id)
        _record_attempt_capture(
            reservation, request, terminal_state, candidate_manifest_ref=manifest_ref, exc=exc,
        )
        raise
    terminal_state = "settled"
    try:
        usage, cost, final = extractor(response)
        usage = dict(usage or {})
        if request.prompt_cache_ttl and not usage.get("prompt_cache_ttl"):
            usage["prompt_cache_ttl"] = request.prompt_cache_ttl
        settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
        _observe_token_density(request, usage)
    except Exception as exc:
        log.exception("Failed to account paid provider response: %s", reservation.attempt_id)
        terminal_state = "dispatched"
        try:
            mark_unresolved(reservation, f"post_response_accounting_failed:{type(exc).__name__}")
            terminal_state = "unresolved"
        except Exception:
            log.exception("Failed to mark post-response accounting failure unresolved")
    _record_attempt_capture(reservation, request, terminal_state, candidate_manifest_ref=manifest_ref)
    return response


# v7 L-C2 split: the one-time legacy usage-telemetry import (source snapshot and
# archive, candidate rows, state-baseline reconciliation, completed watermark)
# lives in ouroboros/usage_legacy_import.py. Re-exported under the historical
# names so callers and monkeypatching tests keep working unchanged (facade
# identity pinned in tests/test_lc2_owner_facades.py).
from ouroboros.usage_legacy_import import (  # noqa: E402, F401 -- intentional public re-exports
    IMPORT_REL,
    _completed_import_watermark,
    _ensure_legacy_imported_locked,
    _legacy_snapshot,
    ensure_legacy_imported,
)
