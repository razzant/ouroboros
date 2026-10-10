"""Durable physical-model-attempt accounting.

The usage store (``usage_store``, ``state/usage.sqlite``) is the monetary
authority: one row per attempt, UPDATEd on each transition, and the summaries
every admission and display reads, maintained in the same transaction.
``llm_usage``/``state.json`` carry attempt-id projections, never another charge
source. Each budget check and its write share one short store transaction;
pricing and network I/O stay outside."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, get_args

from ouroboros.pricing import estimate_cost_optional
from ouroboros._usage_response import (
    _normalized_input_token_usage, _reported_token_count, processing_receipt, observed_processing_mode, usage_from_response,
    _plain, _provider_exception_facts, physical_failure_evidence, provider_failure_payload, provider_cost_value,
)
from ouroboros.review_dispatch import invoke_bound_api_review_paid_stamp
from ouroboros.transport_custody import release_pre_dispatch_attempt
from ouroboros.usage_ledger import (  # noqa: F401 — re-exported substrate
    QUARANTINE_REL, UsageAccountingError, UsageLedgerCorrupt, UsageLockUnavailable,
    _drive_root, _named_lock, _number, _TERMINAL,
)
from ouroboros import usage_store
from ouroboros.usage_store import hold as _locked  # noqa: F401 — the money acquisition primitive
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso  # noqa: F401 -- the accounting module keeps its historical import surface for the L-C2 leaf
from ouroboros._usage_rows import (  # noqa: F401  (re-exported substrate vocabulary)
    REVIEW_ATTRIBUTION_KEYS,
    _breakdown_bucket,
    _physical_call_count,
    breakdown_view,
    _summary,
    _with_integrity,
    _with_limit,
)
from ouroboros.skill_review_usage import skill_review_usage
log = logging.getLogger(__name__)
__all__ = (
    "AttemptRequest", "AttemptReservation", "BudgetExceeded", "PhysicalAttemptCapture",
    "PhysicalAttemptContext", "PhysicalAttemptLimitExceeded", "PhysicalAttemptPreconditionFailed",
    "PhysicalAttemptPreparationFailed",
    "PhysicalAttemptState", "PHYSICAL_ATTEMPT_STATES", "POSITIVE_PHYSICAL_ATTEMPT_STATES",
    "UsageAccountingError", "UsageLedgerCorrupt", "UsageScope", "capture_attempt_ids",
    "bind_physical_attempt_context", "current_physical_attempt_context",
    "current_physical_attempt_predicate", "current_usage_scope",
    "execute_physical_attempt", "execute_physical_attempt_async",
    "last_physical_attempt_capture", "last_root_accounting", "physical_attempt_capture_from_exception",
    "mark_dispatched", "mark_unresolved", "physical_attempt_limit",
    "record_subscription_session",
    "record_unmetered_external_dispatch", "refresh_root_accounting",
    "release_attempt", "reserve_attempt", "settle_attempt",
    "apply_provider_price_receipt", "bind_provider_generation",
    "skill_review_usage", "usage_breakdown", "usage_from_response", "usage_projection", "usage_scope",
    "usage_writer_snapshot", "read_usage_records",
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
_PHYSICAL_DRIVE_ROOT: contextvars.ContextVar[Optional[Tuple[str, pathlib.Path]]] = contextvars.ContextVar("ouroboros_physical_drive_root", default=None)
_ROOT_ACCOUNTING_TELEMETRY: Dict[str, Dict[str, Any]] = {}
_ROOT_ACCOUNTING_TELEMETRY_LOCK = threading.Lock()
_ROOT_ACCOUNTING_TELEMETRY_CAP = 64
_ROOT_RESERVATIONS_KEPT = 8  # identities of the newest appended reservations per root
def _stash_root_accounting(
    root_task_id: str,
    spend: Any,
    root_limit_usd: Optional[float],
    reservation: Optional[Dict[str, Any]] = None,
    *,
    integrity_degraded: bool = False,
) -> None:
    """Refresh the process-local root snapshot. ``spend`` is the rendered
    bucket: its ``settled_usd`` is the known spend money readers decide on, its
    ``accounted_usd`` the exposure including holds; a missing field stays
    unknown (``None``), never borrowed from the other. ``reservation`` is the identity
    of a row this call has just APPENDED (attempt id, task, category, review
    slot): only a successful ``reserve_attempt`` passes one, so a reader that
    finds its own identity here has observed its own reservation — a refresh,
    a settlement or a refused reservation never leaves one."""
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return
    if not isinstance(spend, dict):
        raise TypeError("root accounting snapshot needs a rendered money bucket")
    accounted_usd, settled_usd = _number(spend.get("accounted_usd")), _number(spend.get("settled_usd"))
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
        now = time.monotonic()
        kept = list((_ROOT_ACCOUNTING_TELEMETRY.get(root_task_id) or {}).get("reservations") or [])
        if reservation:
            kept = (kept + [{**reservation, "reserved_monotonic": now}])[-_ROOT_RESERVATIONS_KEPT:]
        _ROOT_ACCOUNTING_TELEMETRY[root_task_id] = {
            "settled_usd": None if settled_usd is None else float(settled_usd),
            "accounted_usd": None if accounted_usd is None else float(accounted_usd),
            "root_limit_usd": None if root_limit_usd is None else float(root_limit_usd),
            # The projection's own integrity verdict rides the snapshot (#1196): a
            # money decision (the exact-pause grant, the Q10 refresh) refuses a
            # degraded tree instead of reading its number as room.
            "integrity_degraded": bool(integrity_degraded),
            "updated_monotonic": now,
            "reservations": kept,
        }

def last_root_accounting(root_task_id: str) -> Optional[Dict[str, Any]]:
    """Newest process-local root snapshot: the KNOWN spend (``settled_usd``)
    limits decide on, the exposure including in-flight holds
    (``accounted_usd``), and the identities of the newest appended
    reservations (each with its own ``age_sec``)."""
    with _ROOT_ACCOUNTING_TELEMETRY_LOCK:
        entry = _ROOT_ACCOUNTING_TELEMETRY.get(str(root_task_id or "").strip())
        if entry is None:
            return None
        entry = dict(entry)
    now = time.monotonic()
    entry["age_sec"] = max(0.0, now - entry.pop("updated_monotonic"))
    reservations = []
    for row in (entry.get("reservations") or []):
        row = dict(row)
        row["age_sec"] = max(0.0, now - float(row.pop("reserved_monotonic", now)))
        reservations.append(row)
    entry["reservations"] = reservations
    return entry

def refresh_root_accounting(
    drive_root: pathlib.Path | str | None,
    root_task_id: str,
    *,
    max_age_sec: float = 0.0,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Refresh a stale root snapshot; on failure return stale/None, never fake $0.

    A DISPLAY reader (``strict=False``) may take the age-bounded cache, or the last
    snapshot when the ledger cannot answer now. A MONEY reader (``strict=True``: the
    exact-pause grant, the Q10 threshold refresh) gets ONE fresh successful observation
    or ``None`` — a snapshot cached before a failed read is unknown spend, not room
    (#1196, the one place that rule lives); a success still refreshes the cache.
    """
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return None
    cached = None if strict else last_root_accounting(root_task_id)
    if cached is not None and max_age_sec > 0 and cached["age_sec"] <= max_age_sec:
        return cached
    try:
        # A ``group:<id>`` key is the whole-work group's accounting (``usage_admission.accounting_key``).
        group = root_task_id[len("group:"):] if root_task_id.startswith("group:") else ""
        projection = usage_projection(drive_root, root_task_id="" if group else root_task_id, billing_group_id=group)
        _stash_root_accounting(
            root_task_id,
            projection,
            _number(projection.get("limit_usd")),
            integrity_degraded=bool(projection.get("integrity_degraded")),
        )
        return last_root_accounting(root_task_id)
    except Exception:
        log.debug("root accounting refresh failed for %s", root_task_id, exc_info=True)
        return cached

def _known_spend_text(summary: Dict[str, Any]) -> str:
    """A refusal's money: the known spend that decided it, the open holds beside it."""
    holds = (_number(summary.get("reserved_usd")) or 0.0) + (_number(summary.get("unresolved_upper_bound_usd")) or 0.0)
    return (f"known=${float(_number(summary.get('settled_usd')) or 0.0):.6f} "
            f"(open holds ${holds:.6f} not counted as spending)")


class BudgetExceeded(UsageAccountingError):
    """Raised before dispatch when known spend has reached an applicable limit."""

    def __init__(self, message: str, *, limit_scope: str = "global", root_task_id: str = "") -> None:
        super().__init__(message)
        self.limit_scope = str(limit_scope or "global")
        self.root_task_id = str(root_task_id or "")


class DispatchFenced(BudgetExceeded):
    """Raised before dispatch while the task is entering an exact budget pause.

    A ``BudgetExceeded`` so every existing catcher treats it as the monetary
    stop it is; ``limit_scope="pausing"`` names the fence. Nothing sent after
    the fence closed can outrun the pause checkpoint (#1196).
    """


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
    claimed_ids: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
@dataclass(frozen=True)
class UsageScope:
    drive_root: pathlib.Path | str | None = None
    task_id: str = ""
    root_task_id: str = ""
    parent_task_id: str = ""
    category: str = "task"
    source: str = "llm"
    # Explicitly attributed system probes/one-shots have no task control owner.
    non_task_operation: bool = False
    review_skill: str = ""
    review_wave_id: str = ""
    review_slot_id: str = ""
    global_limit_usd: Optional[float] = None
    root_limit_usd: Optional[float] = None
    root_cost_ceiling_usd: Optional[float] = None
    global_limit_source: str = ""
    global_limit_revision: Optional[str] = None
    # Whole-work billing group (owner Batch4, ``usage_admission``): empty = this
    # root is its own group under its own cap; a Continue's successor carries the
    # ORIGINAL root's group and that root's original cap.
    billing_group_id: str = ""
    billing_group_limit_usd: Optional[float] = None
    billing_group_limit_source: str = ""
    billing_group_limit_revision: Optional[str] = None
    root_limit_source: str = ""  # Original None + provenance is unlimited, not a new default.
    root_limit_revision: Optional[str] = None
    # The wave a caller named to own its prompt-cache split (skill, plan review). A round the review
    # derives (#1544: retry_key or a fresh id) is attribution only; the split stays per task.
    cache_wave: str = ""
@dataclass(frozen=True)
class PhysicalAttemptContext:
    profile: Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
    rendered_mode: Literal["max", "low", "nano"]
    measurement_basis: Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]
    route_fp: str
    round_id: str
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    context_target_miss: bool
    automatic_pass_used: bool
    # Main's calibration (real tokens per estimated token) of the measurement above;
    # the send finalizer sizes the reply with it. None on rows written before it existed.
    measurement_density: Optional[float] = None
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
    # Route-locality fact (additive): base_url host is localhost/127.0.0.1/::1 (loopback OpenAI-compatible installs — Ollama / LM Studio / vLLM).
    route_is_loopback: bool = False
    # Density uses the fit estimator's count (0 = old producer), while budget
    # reservation retains the conservative raw-base64 estimate above.
    prompt_tokens_bounded_estimate: int = 0
    global_limit_source: str = ""
    global_limit_revision: Optional[str] = None
    processing_preference: str = ""
    submitted_processing_mode: str = ""
    processing_basis: Optional[Dict[str, Any]] = None
    # The same canonical candidate without its Main clock line (``send_clock``);
    # None when the candidate carries none. An identity, never a row field.
    candidate_clock_free_sha256: Optional[str] = None
    effort: Optional[Dict[str, Any]] = None
    allow_live_fetch: bool = True  # False only for a price display: cached tariffs, never a fetch
    provider_receipt_binding: Optional[Dict[str, str]] = None
@dataclass(frozen=True)
class AttemptReservation:
    attempt_id: str
    drive_root: pathlib.Path
    model: str
    provider: str
    reservation_upper_bound_usd: Optional[float]
    processing_preference: str = ""
    submitted_processing_mode: str = ""
    processing_basis: Optional[Dict[str, Any]] = None
    scope: Optional[UsageScope] = None
PhysicalAttemptState = Literal["reserved", "released", "dispatched", "settled", "unresolved"]
PHYSICAL_ATTEMPT_STATES = frozenset(get_args(PhysicalAttemptState))
POSITIVE_PHYSICAL_ATTEMPT_STATES = frozenset({"settled", "dispatched", "unresolved"})
@dataclass(frozen=True)
class PhysicalAttemptCapture:
    attempt_id: str
    model: str
    provider: str
    state: PhysicalAttemptState
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
    route_is_loopback: bool = False  # see AttemptRequest.route_is_loopback
    processing_preference: str = ""
    submitted_processing_mode: str = ""
    processing_basis: Optional[Dict[str, Any]] = None
    effort: Optional[Dict[str, Any]] = None


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
    context: Optional[PhysicalAttemptContext],
    candidate_predicate: Optional[Callable[[AttemptRequest], Any]] = None,
) -> Iterator[Optional[PhysicalAttemptContext]]:
    """Bind frozen Main metadata (None = no Main metadata) and/or a final-fact predicate."""
    if context is not None and not isinstance(context, PhysicalAttemptContext):
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


def current_physical_attempt_drive_root() -> Optional[pathlib.Path]:
    """Execution-local evidence root; not a serialized capture/IPC field."""
    bound, capture = _PHYSICAL_DRIVE_ROOT.get(), _LAST_PHYSICAL_ATTEMPT.get()
    return bound[1] if bound and capture and bound[0] == capture.attempt_id else None


def adopt_physical_attempt_capture(capture: Optional[PhysicalAttemptCapture]) -> None:
    """Carry an accounted worker-thread result into its awaiting caller context.

    This projects an existing receipt; it cannot reserve, settle or forgive an
    attempt. None withdraws a prior capture instead of misattributing the result.
    """
    if capture is not None and not isinstance(capture, PhysicalAttemptCapture):
        raise TypeError("Expected a physical attempt capture")
    _LAST_PHYSICAL_ATTEMPT.set(capture)


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
    except BaseException as exc:
        prior = [str(v) for v in (getattr(exc, "ledger_attempt_ids", None) or []) if v]
        try:
            setattr(exc, "ledger_attempt_ids", list(dict.fromkeys([*prior, *bucket])))
        except Exception: pass
        raise
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
def _claim_physical_dispatch(attempt_id: str = "") -> None:
    state = _PHYSICAL_LIMIT.get()
    if state is None:
        return
    with state.lock:
        if state.used >= state.maximum:
            raise PhysicalAttemptLimitExceeded(f"physical attempt limit exhausted ({state.used}/{state.maximum})")
        state.used += 1
        if attempt_id:
            state.claimed_ids.add(attempt_id)


def _merge_scope(request: AttemptRequest) -> Tuple[AttemptRequest, UsageScope]:
    bound = _CURRENT_SCOPE.get() or UsageScope()
    task_id = str(request.task_id or bound.task_id or "")
    root_task_id = str(request.root_task_id or bound.root_task_id or task_id)
    same_root = root_task_id == str(bound.root_task_id or bound.task_id or "")
    limit_owner = request if request.global_limit_usd is not None else bound
    limit_source = limit_owner.global_limit_source or (
        "attempt_request" if limit_owner is request else "usage_scope"
    )
    scope = UsageScope(
        drive_root=request.drive_root or bound.drive_root,
        task_id=task_id,
        root_task_id=root_task_id,
        parent_task_id=str(request.parent_task_id or (bound.parent_task_id if same_root else "") or ""),
        category=str(request.category or bound.category or "task"),
        source=str(request.source or bound.source or "llm"),
        non_task_operation=bound.non_task_operation and same_root,
        **{key: str(getattr(bound, key, "") or "") for key in REVIEW_ATTRIBUTION_KEYS},
        global_limit_usd=(
            request.global_limit_usd if request.global_limit_usd is not None else bound.global_limit_usd
        ),
        root_limit_usd=(request.root_limit_usd if request.root_limit_usd is not None
                        else bound.root_limit_usd if same_root else None),
        root_limit_source=bound.root_limit_source if same_root else "",
        root_limit_revision=bound.root_limit_revision if same_root else None,
        root_cost_ceiling_usd=bound.root_cost_ceiling_usd if same_root else None,
        global_limit_source=limit_source if limit_owner.global_limit_usd is not None else "",
        global_limit_revision=limit_owner.global_limit_revision if limit_owner.global_limit_usd is not None else None,
        billing_group_id=bound.billing_group_id if same_root else "",
        billing_group_limit_usd=bound.billing_group_limit_usd if same_root else None,
        billing_group_limit_source=bound.billing_group_limit_source if same_root else "",
        billing_group_limit_revision=bound.billing_group_limit_revision if same_root else None,
    )
    if not scope.root_task_id and scope.task_id:
        scope = replace(scope, root_task_id=scope.task_id)
    # Bare AttemptRequest is the ledger primitive; task consumers bind a UsageScope.
    # Only task-bound execution resolves durable lineage here (a synthetic raw
    # ledger request does not manufacture a task result).
    if not scope.non_task_operation and not scope.billing_group_id and bound.task_id and scope.root_task_id and scope.drive_root:
        from ouroboros.usage_admission import task_billing_fields

        scope = replace(scope, **task_billing_fields({"id": scope.task_id}, scope.root_task_id,
                                                    scope.root_limit_usd, scope.drive_root))
    if scope.billing_group_id and scope.drive_root:
        from ouroboros.usage_admission import effective_billing_fields
        scope = replace(scope, **effective_billing_fields(scope.drive_root, scope.root_task_id, {
            key: getattr(scope, key) for key in ("root_limit_usd", "root_limit_source", "root_limit_revision", "billing_group_id",
                "billing_group_limit_usd", "billing_group_limit_source", "billing_group_limit_revision")},
            non_task_operation=scope.non_task_operation))
    if request.root_limit_usd is not None and (scope.root_limit_usd is None or request.root_limit_usd < scope.root_limit_usd):
        scope = replace(scope, root_limit_usd=request.root_limit_usd, root_limit_source="attempt_request", root_limit_revision=None)
    if request.global_limit_usd is None and scope.global_limit_usd is not None:
        request = replace(request, global_limit_usd=scope.global_limit_usd,
                          global_limit_source=scope.global_limit_source,
                          global_limit_revision=scope.global_limit_revision)
    if not request.task_id and scope.task_id:
        # The reservation below keys the task's observed cache split off this id.
        request = replace(request, task_id=scope.task_id, root_task_id=scope.root_task_id)
    return request, scope
from ouroboros._usage_cache_splits import (  # noqa: F401,E402  (re-exported seam)
    invalidate_task_cache_splits, last_task_cache_split,
    reset_task_cache_splits as _reset_task_cache_splits, stash_task_cache_split)


def usage_projection(drive_root=None, *, root_task_id="", global_limit_usd=None,
                     include_roots=False, allow_stale=False, billing_group_id=""):
    """Current read-side projection from the store's summaries; the recorded
    rows retain historical caps. ``include_roots`` (an explicit request) adds
    the per-root map."""
    from ouroboros.usage_admission import current_usage_projection
    return current_usage_projection(drive_root, root_task_id=root_task_id, global_limit_usd=global_limit_usd,
        include_roots=include_roots, allow_stale=allow_stale, billing_group_id=billing_group_id)


def usage_breakdown(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    task_id: str = "", allow_stale: bool = False, include_owners: bool = False,
) -> Dict[str, Any]:
    """Physical-call/token/cost buckets of the global scope, one root or one
    task, read from the store's summaries in one transaction (an addressed
    root+task pair aggregates only its own rows). ``_ledger_high_water_seq`` is
    the ``[epoch, seq]`` publication marker of the SAME read. The global
    per-task/per-root maps enumerate every owner and are rendered only with
    ``include_owners``. ``allow_stale`` is a display read: after the short
    display wait it raises ``UsageLockUnavailable`` (unavailable, never zero)."""
    root = _drive_root(drive_root)
    with usage_store.read(root, allow_stale=allow_stale) as txn:
        result = breakdown_view(txn, root_task_id=root_task_id, task_id=task_id, include_owners=include_owners,
                                degraded=usage_store.integrity_degraded(root))
        result["_ledger_high_water_seq"] = txn.marker()
    return result


def usage_writer_snapshot(
    drive_root: pathlib.Path | str | None = None, *, allow_stale: bool = False,
) -> Dict[str, Any]:
    """The compatibility writer's slim read: the global totals it persists, the
    publication marker, the OpenRouter provider bucket its drift check compares
    and the totals-only money projection, all from ONE store read."""
    root = _drive_root(drive_root)
    with usage_store.read(root, allow_stale=allow_stale) as txn:
        top, openrouter, marker = txn.bucket("global"), txn.bucket("provider", "openrouter"), txn.marker()
        degraded = usage_store.integrity_degraded(root)
    by_provider = {"openrouter": openrouter.render_breakdown()} if openrouter.rows else {}
    for bucket in by_provider.values():
        if degraded:
            _with_integrity(bucket, True)
    return {
        **_with_integrity(top.render_breakdown(), degraded),
        "_ledger_high_water_seq": marker,
        "by_provider": by_provider,
        "_usage_projection": _with_integrity(_with_limit(top.render_summary(), None), degraded),
    }


def read_usage_records(drive_root: pathlib.Path | str | None = None, *, final_only: bool = False) -> list:
    """Every attempt's current row, in write order. Explicit audits and the
    export only (the store keeps no superseded rows, so ``final_only`` is
    implied); ordinary readers use summaries and addressed rows."""
    del final_only
    return usage_store.read_usage_records(_drive_root(drive_root))


def _reservation_cost(request: AttemptRequest) -> Optional[float]:
    explicit = request.max_budget_usd if request.max_budget_usd is not None else request.reservation_usd
    if explicit is not None:
        return _number(explicit)
    if request.force_unknown_reservation:
        return None
    if str(request.provider or "").lower() == "local":
        return 0.0
    # Deliberately the RAW estimate (base64-inclusive), NOT the bounded proxy:
    # for money reservation an over-count on image rounds is the safe
    # direction, while density calibration needs the bounded basis (owner
    # decision 3=A). Unifying the two silently lowers image-round reserves.
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
    cached_tokens = 0
    if cache_write_tokens:
        # Price the task's OWN last observed split, not a full write every round;
        # a missing, stale or other-model split keeps today's full-write reservation.
        cached_tokens = min(prompt_tokens, last_task_cache_split(
            request.task_id, request.model, provider=request.provider,
            processing_mode=request.submitted_processing_mode) or 0)
        cache_write_tokens = prompt_tokens - cached_tokens
    prompt_cache_ttl: Optional[str] = None
    if cache_write_tokens:
        # Price the applied candidate TTL; unknown construction sites use the owner SSOT.
        from ouroboros.config import PROMPT_CACHE_TTL_SCALE, resolve_prompt_cache_ttl

        prompt_cache_ttl = str(request.prompt_cache_ttl or "").strip().lower()
        if prompt_cache_ttl not in PROMPT_CACHE_TTL_SCALE:
            # An inspectable marker-free candidate writes no cache at all. Keep the
            # historical conservative base-tier reservation without misreporting a TTL on
            # the physical settlement; opaque sites still fall back to the owner setting.
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
                     "cached_tokens": cached_tokens,
                     "prompt_cache_ttl": prompt_cache_ttl},
        allow_live_fetch=request.allow_live_fetch,
        provider=request.provider,
        **({"processing_mode": request.submitted_processing_mode}
           if request.submitted_processing_mode else {}),
    )


def _global_limit(request: AttemptRequest) -> float:
    if request.global_limit_usd is not None:
        return max(0.0, float(request.global_limit_usd))
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    configured = resolve_total_budget_usd()
    return float("inf") if configured is None else max(0.0, configured)


_CANDIDATE_ROW_FIELDS = (
    "candidate_raw_sha256", "candidate_raw_size_bytes", "candidate_context_sha256",
    "candidate_context_size_bytes", "candidate_measurement_kind", "physical_context",
    "candidate_manifest_ref",
    "processing_preference", "submitted_processing_mode", "processing_basis",
    "effort",
)


def _check_dispatch_fences(scope: UsageScope, root: pathlib.Path) -> None:
    from ouroboros.budget_pause import budget_fence_selected, dispatch_fenced

    if dispatch_fenced(scope.task_id):
        # Process-local pause fence: no NEW send (loop, tool, reviewer, verdict
        # extraction) under a task that is writing its exact pause checkpoint.
        raise DispatchFenced(
            f"model dispatch fenced: task {scope.task_id} is entering an exact budget pause",
            limit_scope="pausing", root_task_id=scope.root_task_id)
    # The queue's atomic durable root-dispatch fence, read from its snapshot:
    # a fenced root refuses every send until an explicit resume.
    root_task_id = str(scope.root_task_id or "").strip()
    snapshot_path = root / "state" / "queue_snapshot.json"
    if root_task_id and snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise UsageAccountingError(f"root budget fence authority unavailable: {snapshot_path}") from exc
        rows = snapshot.get("budget_root_fences", []) if isinstance(snapshot, dict) else None
        if not isinstance(rows, list):
            raise UsageAccountingError(f"invalid root budget fence authority: {snapshot_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise UsageAccountingError(f"invalid root budget fence row: {snapshot_path}")
            if (str(row.get("root_task_id") or "") != root_task_id or row.get("cause") == "owner_pause"
                    or str(row.get("status") or "") not in {"active", "paused"}):
                continue  # an owner Pause gates launches itself (owner_pause.py): never a money stop
            # ONE member explicitly selected against THIS fence generation is
            # admitted (owner Q9, #1196): the queue recorded that selection on
            # the row itself (a selected hold or an exact Resume handoff, read by
            # the queue's own predicate), and the latch still refuses every
            # unselected member. A just-assigned row may still be published pending.
            selected = False
            for bucket in ("running", "pending"):
                for entry in (snapshot.get(bucket) or []) if isinstance(snapshot, dict) else []:
                    member = entry.get("task") if isinstance(entry, dict) else None
                    if not isinstance(member, dict) or str(member.get("id") or "") != scope.task_id:
                        continue
                    selected = budget_fence_selected(member, row)
            if not selected:
                raise BudgetExceeded(
                    f"root model dispatch paused pending explicit resume for {scope.root_task_id}",
                    limit_scope="root",
                    root_task_id=scope.root_task_id,
                )


def reserve_attempt(request: AttemptRequest) -> AttemptReservation:
    """Atomically check global/root/group limits and record a ``reserved`` attempt whose row carries the applied global limit, its source and revision."""
    request, scope = _merge_scope(request)
    root = _drive_root(scope.drive_root)
    from ouroboros._usage_wait import send_acquisition

    _check_dispatch_fences(scope, root)
    acquire = send_acquisition(lambda: _check_dispatch_fences(scope, root))
    # Pricing I/O stays outside the monetary transaction.
    bound = _reservation_cost(request)
    pricing_known = bound is not None
    attempt_id = uuid.uuid4().hex

    with acquire(root) as view:
        _check_dispatch_fences(scope, root)
        global_limit = _global_limit(request)
        if view.exceeds_limit(global_limit):
            raise BudgetExceeded(
                f"global model budget exhausted: {_known_spend_text(view.summary())}, "
                f"limit=${global_limit:.6f}",
                limit_scope="global",
                root_task_id=scope.root_task_id,
            )
        root_limit: Optional[float] = None
        if scope.root_task_id:  # every rooted attempt refreshes the subtree telemetry, cap or not
            root_summary = view.summary(scope.root_task_id)
            root_limit = None if scope.root_limit_usd is None else max(0.0, float(scope.root_limit_usd))
            _stash_root_accounting(scope.root_task_id, root_summary, root_limit)  # pre-append subtree sum
        if root_limit is not None:
            if view.exceeds_limit(root_limit, root_task_id=scope.root_task_id):
                raise BudgetExceeded(
                    f"root model budget exhausted for {scope.root_task_id}: "
                    f"{_known_spend_text(root_summary)}, limit=${root_limit:.6f}",
                    limit_scope="root",
                    root_task_id=scope.root_task_id,
                )
        from ouroboros.usage_admission import raise_group_refusal, scope_group

        raise_group_refusal(view, scope)
        group_id, group_limit = scope_group(scope)
        view.write(
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
                    "billing_group_id": group_id, "billing_group_limit_usd": group_limit,
                    "billing_group_limit_source": scope.billing_group_limit_source,
                    "billing_group_limit_revision": scope.billing_group_limit_revision,
                    "category": scope.category,
                    **({"non_task_operation": True} if scope.non_task_operation else {}),
                    "source": scope.source,
                    **{key: str(getattr(scope, key, "") or "") for key in REVIEW_ATTRIBUTION_KEYS},
                    # The value checked above, including a resolver fallback, is
                    # the applied limit. A scope snapshot is not a held share.
                    "global_limit_usd": None if global_limit == float("inf") else global_limit,
                    "global_limit_unbounded": global_limit == float("inf"),
                    "global_limit_source": scope.global_limit_source or "settings_budget_resolver",
                    "global_limit_revision": scope.global_limit_revision,
                    "root_limit_usd": scope.root_limit_usd,
                    "root_limit_source": scope.root_limit_source, "root_limit_revision": scope.root_limit_revision,
                    "candidate_raw_sha256": request.candidate_raw_sha256,
                    "candidate_raw_size_bytes": request.candidate_raw_size_bytes,
                    "candidate_context_sha256": request.candidate_context_sha256,
                    "candidate_context_size_bytes": request.candidate_context_size_bytes,
                    "candidate_measurement_kind": request.candidate_measurement_kind,
                    "physical_context": asdict(request.physical_context) if request.physical_context else None,
                    **({"provider_receipt_binding": copy.deepcopy(request.provider_receipt_binding)}
                       if request.provider_receipt_binding else {}),
                    **({"effort": copy.deepcopy(request.effort)} if request.effort is not None else {}),
                    **({"processing_preference": request.processing_preference,
                        "submitted_processing_mode": request.submitted_processing_mode,
                        "processing_basis": copy.deepcopy(request.processing_basis)}
                       if request.processing_preference or request.submitted_processing_mode
                       or request.processing_basis else {}),
                }
        )
        if group_id:
            _stash_root_accounting(f"group:{group_id}",
                                   view.summary(billing_group_id=group_id), group_limit)
        if scope.root_task_id:
            _stash_root_accounting(
                scope.root_task_id,
                view.summary(scope.root_task_id),
                root_limit,
                reservation={
                    "attempt_id": attempt_id, "task_id": scope.task_id,
                    "category": scope.category, "review_slot_id": scope.review_slot_id,
                },
            )
    bucket = _ATTEMPT_COLLECTOR.get()
    if bucket is not None:
        bucket.append(attempt_id)
    return AttemptReservation(attempt_id, root, request.model, request.provider, bound,
                              request.processing_preference, request.submitted_processing_mode,
                              copy.deepcopy(request.processing_basis), scope=scope)


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
    identity = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    attempt_id = f"external-{identity[:24]}"
    from ouroboros.usage_admission import settlement_billing_fields

    real_task = str(task_id or bound.task_id or "")
    real_root = str(root_task_id or (bound.root_task_id if not task_id else "") or real_task)
    billing = settlement_billing_fields(root, real_task, real_root)
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
        "task_id": real_task,
        "root_task_id": real_root,
        "parent_task_id": str(parent_task_id or bound.parent_task_id or ""),
        **{key: value for key, value in billing.items() if key.startswith("billing_group_")},
        "category": str(category or bound.category or "external"),
        "source": str(source or bound.source or "external_skill"),
        "external_dispatch_id_sha256": identity,
    }
    return _record_one_shot(root, row)


def _record_one_shot(root: pathlib.Path, row: Dict[str, Any]) -> str:
    """Idempotently record a one-shot row: an identical identity
    (``usage_store.ONE_SHOT_IDENTITY``) returns the stored attempt, a replay
    under a DIFFERENT identity is a conflict, never a silent overwrite."""
    with _locked(root) as txn:
        if txn.recorded_one_shot(row) is None:
            txn.write(row)
    return str(row["attempt_id"])


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
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cached_tokens: int | None = None,
    reset_at: str = "",
    spend_usd: float | None = None,
    spend_estimated: bool = False,
    credential_profile_id: str = "",
    access_profile: str = "",
    input_token_usage: Dict[str, Any] | None = None,
    review_skill: str = "", review_wave_id: str = "", review_slot_id: str = "",
    attempt_execution: Optional[list[Dict[str, Any]]] = None,
    effort_resolution: Optional[Dict[str, Any]] = None,
) -> str:
    """Record one idempotent session; model observation is not session identity.

    A later model disclosure replays the existing row byte-for-byte, without
    repricing or rewriting it. Custody carries the newly observed actor facts.

    ``input_token_usage`` is the harness's optional NORMALIZED input split
    (total, cache read, cache write). It is validated here, the one place that
    persists it, and it is deliberately outside the idempotent identity: an
    engine that starts reporting it must not rewrite or duplicate a row already
    settled without it. Unreported or unusable stays absent — the legacy
    ``prompt_tokens``/``cached_tokens`` axes keep their own meanings.
    """
    stable_id, route_id = str(session_id or "").strip(), str(route or "").strip()
    if not stable_id or not route_id:
        raise UsageAccountingError("subscription session requires a stable session_id and route")
    bound = _CURRENT_SCOPE.get() or UsageScope()
    root = _drive_root(drive_root or bound.drive_root)
    identity = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    attempt_id = f"session-{identity[:24]}"
    from ouroboros.usage_admission import settlement_billing_fields

    # Explicit custody identities never inherit an unrelated caller's context.
    real_task = str(task_id or bound.task_id or "")
    real_root = str(root_task_id or (bound.root_task_id if not task_id else "") or real_task)
    billing = settlement_billing_fields(root, real_task, real_root)
    input_counters = _normalized_input_token_usage(input_token_usage)
    attribution = {"review_skill": review_skill, "review_wave_id": review_wave_id, "review_slot_id": review_slot_id}
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
        "task_id": real_task,
        "root_task_id": real_root,
        "parent_task_id": str(parent_task_id or bound.parent_task_id or ""),
        **{key: value for key, value in billing.items() if key.startswith("billing_group_")},
        "category": str(category or bound.category or "subagent"),
        "source": str(source or bound.source or "delegated_subagent"),
        **{key: str(attribution[key] or getattr(bound, key, "") or "") for key in REVIEW_ATTRIBUTION_KEYS},
        "subscription_route": route_id,
        "subscription_reset_at": str(reset_at or ""),
        # Empty profile/access means the engine reported none.
        "credential_profile_id": str(credential_profile_id or ""),
        "access_profile": str(access_profile or ""),
        "session_id_sha256": identity,
        **({"effort_resolution": copy.deepcopy(effort_resolution)} if isinstance(effort_resolution, dict) else {}),
        # Present only when the harness reported a complete, valid object.
        **({"input_token_usage": input_counters} if input_counters is not None else {}),
        **({"attempt_execution": copy.deepcopy(attempt_execution)}
           if isinstance(attempt_execution, list) else {}),
        # CPL-5: sessions do not hand the host final wire bytes; disclose the
        # limit (design note §4, provider_side_transform) without a fake seal.
        "model_send_seal": "unobserved",
    }
    return _record_one_shot(root, row)
def _transition(reservation: AttemptReservation, state: str, **fields: Any) -> Dict[str, Any]:
    """One attempt transition in one store transaction: the legality rules
    below, the dispatch recheck of known spend and the fences, then the UPDATE
    with its summary deltas. ``_expected_seq``/``_expected_revision`` make a
    recovery that raced a settlement a no-op returning the current row."""
    from ouroboros._usage_wait import transition_acquisition
    from ouroboros.usage_ledger import is_abandoned_settlement

    acquire = transition_acquisition(state, lambda: _check_dispatch_fences(
        reservation.scope or current_usage_scope() or UsageScope(), reservation.drive_root))
    with (acquire(reservation.drive_root) if acquire else _locked(reservation.drive_root)) as view:
        current = view.attempt(reservation.attempt_id)
        expected_seq = fields.pop("_expected_seq", None)
        expected_revision = fields.pop("_expected_revision", None)
        abandon_reason = fields.pop("_abandon_reason", "")
        if current is None:
            if abandon_reason:
                return {"state": "unknown"}
            raise UsageAccountingError(f"unknown usage attempt {reservation.attempt_id}")
        if ((expected_seq is not None and current.get("seq") != expected_seq)
                or (expected_revision is not None and current.get("revision") != expected_revision)):
            return copy.deepcopy(current)
        if state == "released" and current["state"] == "released":
            return copy.deepcopy(current)
        abandoned = is_abandoned_settlement(current)
        if abandon_reason:
            if current["state"] == "released" or (current["state"] == "settled" and not abandoned):
                return copy.deepcopy(current)  # A real receipt that won the race remains authoritative.
            if current["state"] == "reserved":
                state, fields = "released", {"reason": abandon_reason}
            elif abandoned and fields.get("settle_reason") == "abandoned":
                return copy.deepcopy(current)
            else:
                fields["reason"] = abandon_reason
                if current.get("reason") and not current.get("unresolved_reason"):
                    fields["unresolved_reason"] = current["reason"]
        if state == "unresolved" and abandoned:
            return copy.deepcopy(current)
        if state == "settled" and current["state"] == "settled" and not abandoned:
            if all(current.get(key) == value for key, value in fields.items()):
                return copy.deepcopy(current)
            raise UsageAccountingError(f"conflicting usage settlement: {reservation.attempt_id}")
        if state == "settled" and (abandoned or current["state"] == "unresolved") and fields.get("settle_reason") != "abandoned":
            fields["settle_reason"] = "late_receipt"
        allow_release = bool(fields.pop("_allow_dispatched_release", False))
        if state == "released" and current.get("state") == "dispatched" and not allow_release:
            raise UsageAccountingError("dispatched attempts require a typed pre-dispatch release")
        if state == "dispatched":
            scope = UsageScope(drive_root=reservation.drive_root,
                non_task_operation=current.get("non_task_operation") is True, **{key: current.get(key) for key in
                ("task_id", "root_task_id", "parent_task_id", "root_limit_usd", "root_limit_source", "root_limit_revision", "billing_group_id",
                 "billing_group_limit_usd", "billing_group_limit_source", "billing_group_limit_revision")})
            from ouroboros.owner_pause import member_fence
            if fence := member_fence(scope):
                from ouroboros.llm_attempt import _PhysicalSendNotStarted
                raise _PhysicalSendNotStarted(str(fence.get("reason") or "owner_pause"))
            from ouroboros.usage_admission import effective_billing_fields
            effective = effective_billing_fields(reservation.drive_root, scope.root_task_id, {
                key: getattr(scope, key) for key in ("root_limit_usd", "root_limit_source", "root_limit_revision",
                    "billing_group_id", "billing_group_limit_usd", "billing_group_limit_source", "billing_group_limit_revision")},
                non_task_operation=scope.non_task_operation)
            if scope.root_limit_source == "attempt_request":
                effective.update(root_limit_usd=scope.root_limit_usd, root_limit_source=scope.root_limit_source,
                                 root_limit_revision=scope.root_limit_revision)
            scope = replace(scope, **effective)
            fields.update(effective)
            _check_dispatch_fences(scope, reservation.drive_root)
            limit_request = AttemptRequest(model=reservation.model, provider=reservation.provider,
                global_limit_usd=(current.get("global_limit_usd") if current.get("global_limit_source")
                                  not in {"", "settings_budget_resolver"} else None))
            for axis, limit, identity in (
                ("global", _global_limit(limit_request), None),
                ("root", scope.root_limit_usd, scope.root_task_id),
            ):
                # The reservation's own predicate: known spend reached the limit
                # since this attempt was reserved, so nothing is sent.
                if limit is not None and view.exceeds_limit(limit, root_task_id=identity):
                    raise BudgetExceeded(f"{axis} model budget changed before dispatch", limit_scope=axis,
                                         root_task_id=scope.root_task_id)
            from ouroboros.usage_admission import raise_group_refusal

            raise_group_refusal(view, scope)
        replaced = {"seq", "ts", "revision", "pre_compaction_seq", "settle_reason", "reason"}
        if state == "settled":
            replaced.update(("effort", "effort_resolution", "processing", "speed", "service_tier", "cost_basis", "cost_evidence"))
            if "effort" not in fields and isinstance(current.get("effort"), dict):  # Keep candidate facts, not older observations.
                fields["effort"] = {**current["effort"], "reported": None, "report_source": None}
        row = {key: value for key, value in current.items() if key not in replaced}
        row.update(state=state, **fields)
        if state == "dispatched":
            from ouroboros.owner_pause import OwnerPauseRefused, launch_admission

            try:
                # Money acquisition and preparation precede the final Pause
                # gate. Only the durable local claim holds both locks: the
                # dispatched row COMMITS inside the launch admission.
                with launch_admission(scope):
                    appended = view.write(row, current)
                    view.commit()
            except OwnerPauseRefused as exc:
                from ouroboros.llm_attempt import _PhysicalSendNotStarted
                raise _PhysicalSendNotStarted(str(exc) or "owner_pause") from exc
        else:
            appended = view.write(row, current)
        from ouroboros._usage_money import billing_group_key

        group_id = billing_group_key(current)
        if group_id:
            _stash_root_accounting(f"group:{group_id}",
                view.summary(billing_group_id=group_id),
                _number(current.get("billing_group_limit_usd", current.get("root_limit_usd"))))
        root_task_id = str(current.get("root_task_id") or "")
        if root_task_id:
            _stash_root_accounting(root_task_id, view.summary(root_task_id),
                                   _number(current.get("root_limit_usd")))
        return copy.deepcopy(appended)


def mark_dispatched(
    reservation: AttemptReservation, *,
    candidate_manifest_ref: Optional[Dict[str, Any]] = None,
    local_answer_owner_pid: int = 0,
) -> None:
    invoke_bound_api_review_paid_stamp(fail_closed=True)
    try:
        _claim_physical_dispatch(reservation.attempt_id)
    except PhysicalAttemptLimitExceeded:
        from ouroboros._usage_wait import bounded_cleanup
        try:
            with bounded_cleanup():
                release_attempt(reservation, "physical_attempt_limit", candidate_manifest_ref=candidate_manifest_ref)
        except Exception:
            log.exception("Failed bounded release after physical attempt limit")
        raise
    fields = {"candidate_manifest_ref": candidate_manifest_ref} if candidate_manifest_ref else {}
    if local_answer_owner_pid:
        fields["local_answer_owner_pid"] = local_answer_owner_pid
        from ouroboros.model_wait import current_model_wait
        owner = current_model_wait()
        if (owner is not None and local_answer_owner_pid == os.getpid()
                and owner.task_id == (reservation.scope or UsageScope()).task_id):
            fields.update(owner.bind_answer_consumer())
    _transition(reservation, "dispatched", launch_state="claimed", transport_outcome="unknown", **fields)
    invoke_bound_api_review_paid_stamp(fail_closed=False)


def release_attempt(
    reservation: AttemptReservation, reason: str = "not_dispatched", *, candidate_manifest_ref=None,
    proven_unsent: bool = False,
) -> None:
    _transition(reservation, "released", reason=str(reason or "not_dispatched"),
                _allow_dispatched_release=proven_unsent, **(
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
    expected_seq: Optional[int] = None,
    expected_revision: Optional[int] = None,
) -> str:
    """Close a proven abandoned send without claiming its bound was an actual price.

    The current-state decision and the write share one store transaction; an
    ``expected_revision`` (or the row's ``expected_seq``) that no longer matches
    leaves a concurrent settlement authoritative. A late real receipt may
    replace this administrative settlement; unknown price stays non-final.
    """
    normalized = dict(usage or {})
    measured = any(_reported_token_count(normalized, *keys) for keys in (
        ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")))
    fields = _settlement_fields(reservation, normalized, None, False) if measured else {
        "cost_usd": None, "cost_final": False, "settle_reason": "abandoned"}
    row = _transition(reservation, "settled", _abandon_reason=str(reason or "owner_task_terminal"),
                      _expected_seq=expected_seq, _expected_revision=expected_revision, **fields)
    return str(row["state"])


def settle_attempt(
    reservation: AttemptReservation,
    usage: Optional[Dict[str, Any]] = None,
    *,
    cost_usd: Optional[float] = None,
    cost_final: bool = False,
    expected_revision: Optional[int] = None,
) -> None:
    fields = _settlement_fields(reservation, usage, cost_usd, cost_final)
    _transition(reservation, "settled", _expected_revision=expected_revision, **fields)
    stash_task_cache_split(
        (_CURRENT_SCOPE.get() or UsageScope()).task_id, reservation.model,
        int(fields.get("cached_tokens") or 0), provider=reservation.provider,
        ttl_seconds=3600.0 if fields["prompt_cache_ttl"] == "1h" else 300.0,
        processing_mode=observed_processing_mode(reservation.provider, fields),
    )


def apply_provider_price_receipt(drive_root, attempt_id: str, receipt: dict, *, expected_revision=None) -> dict:
    """Apply a provider-validated exact-attempt fact, preserving other observations.

    Provider leaves prove source bytes before calling this writer. Binding is
    opaque here: only its exact equality with the attempt's binding matters.
    Callers own live task/review custody checks; price evidence grants no right
    to take over an active send or continue its response.
    Final identity/price repeats are duplicates; a different final price conflicts.
    Stale nonfinal revisions never report an applied effect. No network runs here.
    """
    from ouroboros.usage_ledger import provider_price_refinable
    from ouroboros._usage_money import amount, billing_group_key

    root = _drive_root(drive_root)
    cost = provider_cost_value(receipt.get("cost_usd")) if isinstance(receipt, dict) else None
    if (cost is None or receipt.get("attempt_id") != attempt_id
            or not receipt.get("evidence_ref") or not isinstance(receipt.get("binding"), dict)
            or not receipt["binding"]):
        return {"status": "ineligible", "reason": "invalid_price_fact"}
    with _locked(root, migrate=False) as view:
        current = view.attempt(attempt_id)
        binding = (current or {}).get("provider_receipt_binding")
        if (binding is None or receipt["binding"] != binding
                or receipt.get("provider") != current.get("provider")):
            return {"status": "ineligible", "reason": "binding_mismatch"}
        if current.get("cost_final") is True:
            return {"status": "duplicate" if amount(current.get("cost_usd")) == amount(cost) else "conflict", "row": current}
        if not provider_price_refinable(current):
            return {"status": "ineligible", "reason": "attempt_state"}
        if expected_revision is not None and current["revision"] != expected_revision:
            return {"status": "stale", "row": current}
        row = {key: value for key, value in current.items() if key not in {"seq", "ts", "revision", "pre_compaction_seq"}}
        row.update(state="settled", cost_usd=cost, cost_final=True, settle_reason="late_receipt",
                   provider_price_receipt=copy.deepcopy(receipt))
        stored = view.write(row, current)
        for key, summary, limit in (
            (current.get("root_task_id"), view.summary(current.get("root_task_id")), current.get("root_limit_usd")),
            (f"group:{billing_group_key(current)}" if billing_group_key(current) else "",
             view.summary(billing_group_id=billing_group_key(current)), current.get("billing_group_limit_usd")),
        ):
            if key:
                _stash_root_accounting(key, summary, _number(limit))
        return {"status": "applied", "row": stored}


def bind_provider_generation(generation_id: str, *, reservation: Optional[AttemptReservation] = None) -> None:
    """Compatibility entrypoint; the provider leaf owns generation grammar."""
    from ouroboros.openrouter_cost import bind_generation

    bind_generation(generation_id, reservation=reservation)


def _retain_physical_failure(reservation, *, exc=None, response=None):
    """Failure source is durable before a caller can retry or replace its capture."""
    payload = None
    try:
        payload = provider_failure_payload(exc) if exc is not None else _plain(response)
        stream_receipt = (getattr(exc, "stream_receipt", None) if exc is not None else
                          payload.get("_stream_receipt") if isinstance(payload, dict) else None)
        stream_receipt = stream_receipt if isinstance(stream_receipt, dict) else {}
        if not stream_receipt.get("generation_bound"):
            try:
                # Cleanup workers deliberately clear ambient captures. The
                # reservation, and a retained failed stream's ID, still belong
                # to this exact physical send.
                generation_id = stream_receipt.get("generation_id") or (
                    payload.get("id") if isinstance(payload, dict) else None)
                bind_provider_generation(generation_id, reservation=reservation)
                if stream_receipt.get("conflicting_generation_id"):
                    bind_provider_generation(stream_receipt["conflicting_generation_id"], reservation=reservation)
            except Exception:
                log.exception("Failed to bind provider generation: %s", reservation.attempt_id)
        facts = physical_failure_evidence(reservation.drive_root, reservation.attempt_id, payload=payload, exc=exc, response=response)
        if facts is not None:
            with _locked(reservation.drive_root) as view:
                row = view.attempt(reservation.attempt_id)
                view.record_evidence(reservation.attempt_id, {"physical_failure": facts}, expected_revision=row["revision"])
    except Exception:
        log.exception("Failed to retain physical failure evidence: %s", reservation.attempt_id)
    return payload


def _settlement_fields(reservation, usage, cost_usd, cost_final) -> Dict[str, Any]:
    """Normalize reported usage before the monetary transaction; no live pricing I/O."""
    normalized = dict(usage or {})
    receipt = processing_receipt(reservation.provider, normalized,
                                 requested=reservation.processing_preference,
                                 submitted_native=reservation.submitted_processing_mode)
    if receipt is not None:
        normalized["processing"] = receipt
    prompt_tokens = _reported_token_count(normalized, "prompt_tokens", "input_tokens")
    completion_tokens = _reported_token_count(normalized, "completion_tokens", "output_tokens")
    cached_tokens = _reported_token_count(normalized, "cached_tokens")
    cache_write_tokens = _reported_token_count(normalized, "cache_write_tokens")
    cost = _number(cost_usd)
    pricing_mode = observed_processing_mode(reservation.provider, normalized)
    has_usage = bool((prompt_tokens or 0) or (completion_tokens or 0))
    if cost is None and str(reservation.provider or "").lower() == "local":
        cost, cost_final = 0.0, True
    elif cost is None and cost_usd is None and not normalized.get("cost_invalid") and has_usage:
        cost = estimate_cost_optional(
            reservation.model,
            int(prompt_tokens or 0),
            int(completion_tokens or 0),
            cache_usage={"cached_tokens": int(cached_tokens or 0),
                         "cache_write_tokens": int(cache_write_tokens or 0),
                         "cache_write_tokens_by_ttl": normalized.get("cache_write_tokens_by_ttl"),
                         "prompt_cache_ttl": str(normalized.get("prompt_cache_ttl") or "")},
            allow_live_fetch=False,
            provider=reservation.provider,
            **({"processing_mode": pricing_mode} if pricing_mode else {}),
        )
        cost_final = False
    return dict(
        cost_usd=cost,
        cost_final=bool(cost_final and cost is not None),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        prompt_cache_ttl=str(normalized.get("prompt_cache_ttl") or ""),
        **{key: copy.deepcopy(normalized[key]) for key in ("effort", "effort_resolution", "processing", "speed", "service_tier", "cost_basis", "cost_evidence")
           if key in normalized},
    )


def _observe_token_density(request: AttemptRequest, usage: Optional[Dict[str, Any]]) -> None:
    # Lives beside the density store (capability_evidence) since this module's
    # size ceiling; the settlement paths keep this historical seam name.
    from ouroboros.capability_evidence import observe_token_density

    observe_token_density(request, usage, drive_root_resolver=_drive_root)


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


def _return_physical_claim(attempt_id: str) -> None:
    """Return exactly this actor's proven-unsent claim; paid stamps are separate."""
    state = _PHYSICAL_LIMIT.get()
    if state is not None:
        with state.lock:
            if attempt_id in state.claimed_ids:
                state.claimed_ids.remove(attempt_id)
                state.used -= 1


def _terminalize_failed_attempt(reservation: AttemptReservation, exc: BaseException) -> str:
    """Route a raised provider send to its honest terminal ledger state."""
    if getattr(exc, "receiver_abandoned", False):  # owner Pause: the sender settles it late; still dispatched
        return "dispatched"
    payload = _retain_physical_failure(reservation, exc=exc)
    if release_pre_dispatch_attempt(reservation, exc):
        _return_physical_claim(reservation.attempt_id)
        return "released"
    provider = str(reservation.provider or "").strip().lower()
    stream_usage = getattr(exc, "stream_usage", None)
    for evidence in ({"usage": stream_usage}, payload):
        if isinstance(evidence, dict):
            evidence = {key: value for key, value in evidence.items() if key != "error"}
        usage, cost, final = usage_from_response(evidence)
        if final and cost is not None:
            settle_attempt(reservation, usage, cost_usd=cost, cost_final=True)
            return "settled"
    if provider == "openrouter":
        for reason, predicate in (("pre_routing_rejection", _is_pre_routing_rejection), ("tos_rejection", _is_tos_rejection)):
            if predicate(exc):
                _transition(reservation, "settled", cost_usd=0.0, cost_final=True, settle_reason=reason)
                return "settled"
    if isinstance(stream_usage, dict) and stream_usage:
        # The usage frame was read before the body was judged unusable: money is
        # known, so settle through the success path's extractor and cost derivation
        # from that frame alone (no assembled-body facts such as service_tier, no
        # cache-TTL injection or token-density observation); only the answer is missing.
        usage, cost, final = usage_from_response({"usage": stream_usage})
        settle_attempt(reservation, dict(usage or {}), cost_usd=cost, cost_final=final)
        return "settled"
    else:
        from ouroboros.transport_custody import attempt_custody_event_fields
        cause = attempt_custody_event_fields(exc).get("transport_cause_type")
        suffix = f" [cause: {cause}]" if cause else ""
        # Suffix leads: a verbose provider body must not truncate it away (mark_unresolved keeps 500 chars).
        mark_unresolved(reservation, f"{type(exc).__name__}{suffix}: {exc}")
        return "unresolved"


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
        route_is_loopback=bool(request.route_is_loopback),
        processing_preference=request.processing_preference,
        submitted_processing_mode=request.submitted_processing_mode,
        processing_basis=copy.deepcopy(request.processing_basis),
        effort=copy.deepcopy(request.effort),
    )
    _LAST_PHYSICAL_ATTEMPT.set(capture)
    _PHYSICAL_DRIVE_ROOT.set((reservation.attempt_id, reservation.drive_root))
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
    from ouroboros._usage_wait import bounded_cleanup

    manifest_ref = getattr(exc, "candidate_manifest_ref", None) or candidate_manifest_ref
    capture_state = "reserved"
    # This function is called only before send(): returning the local claim is
    # justified even if the ledger cannot record release. The bound stays held.
    _return_physical_claim(reservation.attempt_id)
    try:
        with bounded_cleanup():
            release_attempt(reservation, f"before_dispatch_failed:{type(exc).__name__}",
                            candidate_manifest_ref=manifest_ref, proven_unsent=True)
        capture_state = "released"
    except Exception:
        log.exception("Failed to release pre-dispatch attempt: %s", reservation.attempt_id)
    failure = exc if isinstance(exc, (PhysicalAttemptPreparationFailed, PhysicalAttemptLimitExceeded, BudgetExceeded, asyncio.CancelledError)) else PhysicalAttemptPreparationFailed(
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
    late_owner: Optional[Callable[[Any, Optional[BaseException]], None]] = None,
) -> Any:
    """Execute one synchronous provider send with durable lifecycle accounting."""
    _LAST_PHYSICAL_ATTEMPT.set(None)
    reservation = reserve_attempt(request)
    manifest_ref = None
    try:
        manifest_ref = before_dispatch(reservation) if before_dispatch is not None else None
        if manifest_ref is not None and not isinstance(manifest_ref, dict):
            raise TypeError("before_dispatch must return a manifest ref object or None")
        # Capture/serialization is preparation. Dispatch gates after acquisition.
        prepared_capture = _record_attempt_capture(reservation, request, "reserved", candidate_manifest_ref=manifest_ref)
        dispatch_capture = replace(prepared_capture, state="dispatched")
        mark_dispatched(reservation, candidate_manifest_ref=manifest_ref, local_answer_owner_pid=os.getpid())
        _LAST_PHYSICAL_ATTEMPT.set(dispatch_capture)
    except BaseException as exc:
        failure = _pre_dispatch_failure(
            reservation, request, exc, candidate_manifest_ref=manifest_ref)
        if failure is exc:
            raise
        raise failure from exc
    try:
        from ouroboros._usage_wait import late_settler, model_send
        response = model_send(reservation, send, settle_late=late_settler(
            reservation, request, extractor, manifest_ref, dispatch_capture, late_owner))
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
    return _account_response(reservation, request, response, extractor, manifest_ref)


def _account_response(reservation, request, response, extractor, manifest_ref):
    """Complete received-response accounting once, preserving its open bound on failure."""
    payload = _retain_physical_failure(reservation, response=response)
    terminal_state = "settled"
    try:
        usage, cost, final = extractor(response, payload=payload) if extractor is usage_from_response else extractor(response)
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
    late_owner: Optional[Callable[[Any, Optional[BaseException]], None]] = None,
) -> Any:
    _LAST_PHYSICAL_ATTEMPT.set(None)
    from ouroboros._usage_wait import model_send_async, postresponse_off_loop, presend_off_loop, retain_unadopted

    reservation = await presend_off_loop(reserve_attempt, request, on_cancel=lambda held: (
        _pre_dispatch_failure(held, request, asyncio.CancelledError())))
    manifest_ref = None
    try:
        if before_dispatch is not None:
            pending_manifest = before_dispatch(reservation)
            manifest_ref = await pending_manifest if hasattr(pending_manifest, "__await__") else pending_manifest
        if manifest_ref is not None and not isinstance(manifest_ref, dict):
            raise TypeError("before_dispatch must return a manifest ref object or None")
        prepared_capture = _record_attempt_capture(reservation, request, "reserved", candidate_manifest_ref=manifest_ref)
        dispatch_capture = replace(prepared_capture, state="dispatched")
        await presend_off_loop(mark_dispatched, reservation, candidate_manifest_ref=manifest_ref,
                               local_answer_owner_pid=os.getpid())
        _LAST_PHYSICAL_ATTEMPT.set(dispatch_capture)
    except BaseException as exc:
        failure = await presend_off_loop(_pre_dispatch_failure,
            reservation, request, exc, candidate_manifest_ref=manifest_ref)
        if failure is exc:
            raise
        raise failure from exc
    async def complete():
        try:
            response = await send()
        except BaseException as exc:
            terminal_state = "dispatched"
            try:
                terminal_state = await presend_off_loop(_terminalize_failed_attempt, reservation, exc)
            except Exception:
                log.exception("Failed to mark provider attempt unresolved: %s", reservation.attempt_id)
            _record_attempt_capture(
                reservation, request, terminal_state, candidate_manifest_ref=manifest_ref, exc=exc,
            )
            raise
        return await postresponse_off_loop(_account_response, reservation, request, response, extractor, manifest_ref,
                                           retain_on_cancel=retain_unadopted(request, reservation))


    try:
        return await model_send_async(reservation, complete, retain_on_cancel=retain_unadopted(request, reservation),
            retain_on_abandon=retain_unadopted(request, reservation, control_reason="owner_pause_abandoned"),
            late_owner=late_owner)
    except BaseException as exc:
        if getattr(exc, "receiver_abandoned", False):
            _record_attempt_capture(reservation, request, "dispatched", candidate_manifest_ref=manifest_ref, exc=exc)
        elif getattr(exc, "model_sender_cancelled_before_entry", False):
            await presend_off_loop(_pre_dispatch_failure, reservation, request, exc, candidate_manifest_ref=manifest_ref)
        elif getattr(exc, "model_sender_not_started", False):
            terminal_state = await presend_off_loop(_terminalize_failed_attempt, reservation, exc)
            _record_attempt_capture(reservation, request, terminal_state,
                                    candidate_manifest_ref=manifest_ref, exc=exc)
        raise



# Read-side admission projections over this ledger (the review wave's fit and the
# whole-work group axis) live in their own leaf; the historical binding stays here.
from ouroboros.usage_admission import review_wave_admission  # noqa: E402,F401 -- historical import surface
