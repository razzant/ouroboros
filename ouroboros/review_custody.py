"""Small process-local custody seam for review workers.

The review substrate owns parsing and quorum semantics.  This module owns only
the physical-worker lifecycle: a logical slot deadline may return a typed
in-flight actor, while a retry with the same caller identity discovers the
existing worker or consumes its late result instead of dispatching twice.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from ouroboros.deadline_utils import review_operation_timeout_sec
from ouroboros.observability import new_call_id
from ouroboros.review_dispatch import slot_id_for_row
from ouroboros.usage_accounting import (
    PHYSICAL_ATTEMPT_STATES, POSITIVE_PHYSICAL_ATTEMPT_STATES,
)
from ouroboros.utils import emit_cognitive_operation_event

log = logging.getLogger("review_custody")


@dataclass
class ActiveReviewAttempt:
    key: str
    operation_id: str
    event: threading.Event = field(default_factory=threading.Event)
    actor: Any = None
    timed_out: bool = False
    retry_state: Dict[str, Any] = field(default_factory=dict)
    pending_invocation_checkpoint: Callable[[str], None] | None = None


_ACTIVE_LOCK = threading.Lock()
_ACTIVE: Dict[str, ActiveReviewAttempt] = {}
# A provider response whose physical outcome is unknown may never be bought a
# second time under the same logical identity.  Keep only the compact identity,
# not the full actor/prompt, until process exit.
_NO_RESEND: Dict[str, str] = {}
_PENDING_STATES = {"in_flight", "custody_lost"}
_PHYSICAL_CAPTURE_STATES = frozenset({"", *POSITIVE_PHYSICAL_ATTEMPT_STATES})
_POSITIVE_CAPTURE_STATES = POSITIVE_PHYSICAL_ATTEMPT_STATES


@dataclass
class _ReviewAttemptHistory:
    """The strongest physical fact observed across one actor's retry rail."""

    dispatched: bool = False
    capture_state: str = ""
    provider_status_code: Optional[int] = None
    unknown_outcome_seen: bool = False

    def observe(self, error: Any = None) -> None:
        capture = getattr(error, "physical_attempt_capture", None)
        if capture is None:
            try:
                from ouroboros.usage_accounting import last_physical_attempt_capture

                capture = last_physical_attempt_capture()
            except Exception:
                capture = None
        state = str(getattr(capture, "state", "") or "").strip().lower()
        if state and state not in PHYSICAL_ATTEMPT_STATES:
            # A malformed non-empty capture cannot prove that no request was
            # admitted.  Preserve the strongest safe interpretation across the
            # whole retry rail, so a later exception/status cannot authorize a
            # second paid send.
            self.dispatched = True
            self.capture_state = "unresolved"
            self.unknown_outcome_seen = True
            self.provider_status_code = None
            return
        if state not in _POSITIVE_CAPTURE_STATES:
            return
        self.dispatched = True
        self.capture_state = state
        status = getattr(capture, "provider_status_code", None)
        if state in {"dispatched", "unresolved"} and not (
            isinstance(status, int) and not isinstance(status, bool)
            and 400 <= status <= 599
        ):
            self.unknown_outcome_seen = True
        elif isinstance(status, int) and not isinstance(status, bool):
            self.provider_status_code = status

    def preserve(self, failure_custody: Dict[str, Any], current_state: str) -> None:
        if not self.dispatched or current_state not in {"", "reserved", "released"}:
            return
        failure_custody["physical_attempt_state"] = (
            "unresolved" if self.unknown_outcome_seen else self.capture_state or "unresolved"
        )
        if self.provider_status_code is not None and not self.unknown_outcome_seen:
            failure_custody["provider_status_code"] = self.provider_status_code
        elif self.unknown_outcome_seen:
            failure_custody.pop("provider_status_code", None)


def _review_exception_projection(
    exc: BaseException,
    executor_custody: Any,
    history: _ReviewAttemptHistory,
    retry_state: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], str, Optional[int], str, str]:
    """Project one failed actor while retaining earlier rail custody."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.usage_accounting import BudgetExceeded, UsageAccountingError

    failure_custody = dict(executor_custody or {})
    capture = getattr(exc, "physical_attempt_capture", None)
    capture_state = str(getattr(capture, "state", "") or "").strip().lower()
    malformed_capture = bool(
        capture_state and capture_state not in PHYSICAL_ATTEMPT_STATES
    )
    if malformed_capture:
        # Unknown capture values are custody loss, even when the exception also
        # carries an HTTP status.  The status cannot repair malformed physical
        # provenance, and retaining it would make the row appear replayable.
        capture_state = "unresolved"
        failure_custody["physical_attempt_state"] = capture_state
        failure_custody.pop("provider_status_code", None)
        history.dispatched = True
        history.capture_state = capture_state
        history.unknown_outcome_seen = True
        history.provider_status_code = None
    if capture_state:
        failure_custody["physical_attempt_state"] = capture_state
    http_status = next((value for value in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(capture, "provider_status_code", None),
        failure_custody.get("provider_status_code"),
    ) if isinstance(value, int) and not isinstance(value, bool)), None)
    if http_status is not None and not malformed_capture:
        failure_custody["provider_status_code"] = http_status
    elif malformed_capture:
        http_status = None
        failure_custody.pop("provider_status_code", None)
    if int(failure_custody.get("native_rounds") or 0) > 0 and not history.dispatched:
        # The executor's own proven fact: a native episode that ran paid rounds
        # was dispatched whatever the exception's capture says — a mid-episode
        # deadline or transport end must never read as a $0 retryable row. The
        # rounds that ran are SETTLED ledger rows; only an exception's own
        # positive capture may say the last send's outcome is unknown.
        history.dispatched = True
        history.capture_state = "settled"
    history.preserve(failure_custody, capture_state)
    dispatched = history.dispatched
    if history.unknown_outcome_seen:
        http_status = None
    if (
        history.provider_status_code is not None
        and not history.unknown_outcome_seen
        and capture_state in {"", "reserved", "released"}
    ):
        http_status = history.provider_status_code
    physical_state = str(failure_custody.get("physical_attempt_state") or "").strip().lower()
    route_state = ""
    if not dispatched and isinstance(exc, ReviewRouteUnavailable):
        route_state = _worker_exception_operation_state(
            exc, {**(retry_state or {}), **failure_custody},
        )
    operation_state = (
        "custody_lost" if history.unknown_outcome_seen else route_state
        or ("not_dispatched" if not dispatched and (
            capture_state in {"reserved", "released"}
            or isinstance(exc, BudgetExceeded)
            or str(getattr(exc, "code", "") or "") == "deadline_exhausted"
        ) else "settled")
    )
    failure_code = (
        "provider_outcome_unknown"
        if dispatched and physical_state in {"dispatched", "unresolved"}
        and not isinstance(http_status, int)
        else str(getattr(exc, "code", "") or "")
    )
    # Origin is a producer fact, not the rendered transport label (which can
    # contain error prose). Ledger/admission and authority refusals stay
    # independent of the owner's advisory review enforcement.
    if isinstance(exc, UsageAccountingError) or failure_code in {
        "subscription_window_exhausted", "credential_pool_exhausted",
    }:
        phase = "admission"
    elif failure_code == "deadline_exhausted":
        phase = "deadline"
    elif isinstance(exc, ReviewRouteUnavailable) and failure_code not in {
        "api_chat_unavailable", "route_unavailable", "harness_unavailable",
        "provider_unavailable", "native_inspection_unavailable",
        "native_bound_below_first_send", "native_round_without_progress",
        "native_transcript_cap_exceeded",
    }:
        phase = "authority"
    else:
        phase = "delivery"
    failure_custody["review_failure_phase"] = phase
    return failure_custody, capture_state, http_status, operation_state, failure_code


def _worker_exception_operation_state(
    exc: BaseException, retry_state: Dict[str, Any],
) -> str:
    """Project a worker exception onto custody without guessing from prose.

    Route construction and admission can fail before a physical attempt exists.
    Those typed refusals are retryable $0 rows. A pending invocation, a started
    delegated run, or a positive physical capture is already custody-bearing,
    so it must not be turned into a fresh retry.
    """
    code = str(getattr(exc, "code", "") or "")
    capture = getattr(exc, "physical_attempt_capture", None)
    capture_state = str(getattr(capture, "state", "") or "").strip().lower()
    if capture_state and capture_state not in PHYSICAL_ATTEMPT_STATES:
        return "custody_lost"
    if capture_state in _POSITIVE_CAPTURE_STATES:
        return "settled"
    if bool(getattr(exc, "delegated_run_started", False)) or str(
        getattr(exc, "delegated_run_id", "") or ""
    ).strip():
        return "settled"
    try:
        from ouroboros.review_execution import ReviewRouteUnavailable
    except Exception:
        ReviewRouteUnavailable = ()  # type: ignore[assignment]
    if isinstance(exc, ReviewRouteUnavailable):
        if code == "review_custody_lost" or str(
            retry_state.get("pending_invocation_id") or ""
        ).strip():
            return "custody_lost"
        # Only refusals emitted before the route's paid write-ahead boundary
        # are retryable $0. Later custody/checkpoint failures remain settled:
        # their absence of provider-capture metadata cannot erase the stamp.
        if code in {
            "api_chat_unavailable", "route_unavailable", "harness_unavailable",
            "provider_unavailable", "deadline_exhausted",
            "subscription_window_exhausted",
            "credential_pool_exhausted", "session_task_missing",
            "session_target_unparsable", "session_route_unconfigured",
            "custody_root_missing", "session_root_missing",
            "unknown_review_route", "review_route_not_implemented",
            # Native tool-round refusals raised BEFORE the first provider send.
            "native_inspection_unavailable", "native_bound_below_first_send",
        }:
            return "not_dispatched"
        return "settled"
    if code == "deadline_exhausted":
        return "not_dispatched"
    # A subscription/credential window refusal from route health is also before
    # POST. Once a delegated run exists, the started-run branch above wins.
    if code in {"subscription_window_exhausted", "credential_pool_exhausted"}:
        return "not_dispatched"
    return "settled"


def _attach_worker_exception_facts(
    actor: Any, exc: BaseException, retry_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Carry typed capture/status facts onto a synthetic worker actor."""
    if isinstance(actor, dict):
        usage = dict(actor.get("usage") or {})
    else:
        usage = dict(getattr(actor, "usage", None) or {})
    capture = getattr(exc, "physical_attempt_capture", None)
    capture_state = str(getattr(capture, "state", "") or "").strip().lower()
    if capture_state:
        usage["physical_attempt_state"] = capture_state
    for key in ("pending_invocation_id", "delegated_run_id"):
        value = str((retry_state or {}).get(key) or "").strip()
        if value and key not in usage:
            usage[key] = value
    status = getattr(capture, "provider_status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        usage["provider_status_code"] = status
        if not isinstance(actor, dict) and getattr(actor, "http_status", None) is None:
            actor.http_status = status
    if isinstance(actor, dict):
        actor["usage"] = usage
        code = str(getattr(exc, "code", "") or "")
        if code:
            actor["failure_code"] = code
        reset_at = str(getattr(exc, "reset_at", "") or "")
        if reset_at:
            actor["reset_at"] = reset_at
    else:
        actor.usage = usage
        code = str(getattr(exc, "code", "") or "")
        if code and not str(getattr(actor, "failure_code", "") or ""):
            actor.failure_code = code
        reset_at = str(getattr(exc, "reset_at", "") or "")
        if reset_at and not str(getattr(actor, "reset_at", "") or ""):
            actor.reset_at = reset_at


def _terminal_provider_status(actor: Any, failure_custody: Dict[str, Any]) -> int | None:
    """Return a typed terminal HTTP status carried by this actor only.

    A dispatched/unresolved capture is safe to replay inside its exact logical
    cycle only when the provider returned a terminal HTTP response. Never use
    an unrelated ContextVar or exception text as proof of that response.
    """
    candidates = (
        getattr(actor, "http_status", None),
        failure_custody.get("provider_status_code"),
        failure_custody.get("http_status"),
    )
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            status = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 400 <= status <= 599:
            return status
    return None


def _row_is_pending(row: Dict[str, Any]) -> bool:
    return bool(row.get("late_result_pending")) or str(
        row.get("operation_state") or ""
    ) in _PENDING_STATES


def _custody_lost_row(
    surface: str, index: Any, reason: str, row: Any = None,
) -> Dict[str, Any]:
    lost = copy.deepcopy(row) if isinstance(row, dict) else {}
    if not str(lost.get("slot_id") or lost.get("slot") or ""):
        try:
            position = int(index) + 1
        except (TypeError, ValueError):
            position = 0
        prefix = (
            "custody_lost_scope_slot"
            if surface == "scope_review"
            else "custody_lost_slot"
        )
        lost["slot_id"] = slot_id_for_row(position, prefix=prefix)
    lost.setdefault("status", "error")
    lost.setdefault("error", reason)
    lost["operation_state"] = "custody_lost"
    lost["late_result_pending"] = True
    return lost


def _freeze_roster_rows(usage_ctx: Any, surface: str, raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        rows: List[Any] = []
    elif isinstance(raw, (list, tuple)):
        rows = list(raw)
    else:
        rows = [_custody_lost_row(surface, "container", "review roster is not a list")]
        setattr(usage_ctx, "_review_custody_lost", True)
    by_slot: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = copy.deepcopy(value) if isinstance(value, dict) else _custody_lost_row(
            surface, index, "review roster row is not an object",
        )
        slot_id = str(row.get("slot_id") or row.get("slot") or "")
        if not slot_id:
            row = _custody_lost_row(surface, index, "review roster row has no slot id", row)
            slot_id = str(row["slot_id"])
        if (
            not isinstance(value, dict)
            or slot_id in by_slot
            or str(row.get("operation_state") or "") == "custody_lost"
        ):
            setattr(usage_ctx, "_review_custody_lost", True)
        if slot_id in by_slot:
            by_slot[slot_id] = _custody_lost_row(
                surface, index, "review roster contains duplicate slot ids", by_slot[slot_id],
            )
            continue
        by_slot[slot_id] = row
    return by_slot


def prepare_frozen_review_reconciliation(usage_ctx: Any, attempt: Any) -> None:
    """Expose one durable commit attempt as the exact reconciliation roster."""
    triad = _freeze_roster_rows(
        usage_ctx, "multi_model_review", getattr(attempt, "triad_raw_results", None),
    )
    raw_scope = getattr(attempt, "scope_raw_result", None)
    scope_wrapper = copy.deepcopy(raw_scope) if isinstance(raw_scope, dict) else {}
    if isinstance(raw_scope, dict):
        scope_input = (
            raw_scope.get("raw_results")
            if "raw_results" in raw_scope
            else [raw_scope] if raw_scope else []
        )
    else:
        scope_input = raw_scope
    scope = _freeze_roster_rows(usage_ctx, "scope_review", scope_input)
    frozen = {"multi_model_review": triad, "scope_review": scope}
    if not triad and not scope:
        empty_row = _custody_lost_row(
            "multi_model_review", "empty", "review roster is empty",
        )
        triad[str(empty_row["slot_id"])] = empty_row
        setattr(usage_ctx, "_review_custody_lost", True)
    setattr(usage_ctx, "_review_frozen_rows", frozen)
    setattr(usage_ctx, "_review_frozen_scope_raw", scope_wrapper)


def reconcile_reserved_review_roster(usage_ctx: Any, reserved: Any) -> None:
    """Reconcile a first paid wave against its write-ahead roster.

    The durable reservation is the same actor authority used by a later retry.
    Reusing the frozen-roster merge here prevents an executor exception from
    silently dropping a reserved operation before the first caller returns.
    """
    if not isinstance(reserved, dict):
        return
    prepare_frozen_review_reconciliation(
        usage_ctx,
        SimpleNamespace(
            triad_raw_results=copy.deepcopy(
                reserved.get("multi_model_review") or []
            ),
            scope_raw_result={
                "raw_results": copy.deepcopy(reserved.get("scope_review") or [])
            },
        ),
    )
    merge_frozen_review_reconciliation(usage_ctx)


def _frozen_actor(row: Dict[str, Any], slot: Any) -> Any:
    from ouroboros.review_substrate import ReviewActorRecord

    status = str(row.get("status") or "")
    raw_text = str(row.get("raw_text") or row.get("text") or "")
    actor_status = (
        "not_dispatched" if status == "not_dispatched"
        else "error" if status == "error"
        else "ok" if raw_text or status in {"responded", "ok", "empty", "parse_failure", "partial"}
        else "error"
    )
    usage = copy.deepcopy(row.get("usage") or {}) if isinstance(row.get("usage"), dict) else {}
    for key in (
        "pending_invocation_id", "delegated_run_id", "physical_attempt_state",
        "provider_status_code", "delegated_run_started", "review_thread_id",
        "review_turn_id", "review_thread_receipt", "auth_route_receipt",
        "profile_continuity_receipt", "applied_profile", "capability_delta",
    ):
        value = row.get(key)
        if value not in (None, "") and key not in usage:
            usage[key] = copy.deepcopy(value)
    if row.get("failure_phase"):
        usage["review_failure_phase"] = str(row["failure_phase"])
    operation_state = str(row.get("operation_state") or "settled")
    http_status = row.get("http_status")
    if isinstance(http_status, bool):
        http_status = None
    elif http_status is not None:
        try:
            http_status = int(http_status)
        except (TypeError, ValueError, OverflowError):
            http_status = None
    if http_status is not None and "http_status" not in usage:
        usage["http_status"] = http_status
    pending = bool(row.get("late_result_pending")) or operation_state in _PENDING_STATES
    return ReviewActorRecord(
        slot_id=str(getattr(slot, "slot_id", "") or ""),
        model=str(row.get("model_id") or row.get("model") or getattr(slot, "model", "")),
        status=actor_status,
        raw_text=raw_text,
        error=str(row.get("error") or ""),
        usage=usage,
        prompt_ref=dict(row.get("prompt_ref") or {}),
        response_ref=dict(row.get("response_ref") or {}),
        operation_id=str(row.get("operation_id") or ""),
        late_result_pending=pending,
        transport_status=str(row.get("transport_status") or ""),
        failure_code=str(row.get("failure_code") or ""),
        reset_at=str(row.get("reset_at") or ""),
        http_status=http_status,
        parse_status=str(row.get("parse_status") or ""),
        semantic_verdict=str(row.get("semantic_verdict") or ""),
        provider=str(row.get("provider") or ""),
        actor_role=str(row.get("actor_role") or ""),
        coverage=copy.deepcopy(row.get("coverage") or {}) if isinstance(row.get("coverage"), dict) else {},
        quorum_contribution=bool(row.get("quorum_contribution")),
        reason=str(row.get("reason") or ""),
        enforcement_impact=str(row.get("enforcement_impact") or ""),
        operation_state=operation_state,
    )


def merge_frozen_review_reconciliation(usage_ctx: Any) -> None:
    """Keep the original roster; only its exact pending operation may change."""
    frozen = getattr(usage_ctx, "_review_frozen_rows", None)
    if not isinstance(frozen, dict):
        return

    def _merge(surface: str, current: List[Any]) -> List[Dict[str, Any]]:
        originals = frozen.get(surface) if isinstance(frozen.get(surface), dict) else {}
        current_groups: Dict[str, List[Dict[str, Any]]] = {}
        invalid_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(current):
            if not isinstance(row, dict):
                invalid_rows.append(_custody_lost_row(
                    surface, index, "current review roster row is not an object",
                ))
                setattr(usage_ctx, "_review_custody_lost", True)
                continue
            slot_id = str(row.get("slot_id") or row.get("slot") or "")
            if not slot_id:
                invalid_rows.append(_custody_lost_row(
                    surface, index, "current review roster row has no slot id", row,
                ))
                setattr(usage_ctx, "_review_custody_lost", True)
                continue
            current_groups.setdefault(slot_id, []).append(row)
        ambiguous_slots = {
            slot_id for slot_id, rows in current_groups.items() if len(rows) != 1
        }
        if ambiguous_slots:
            setattr(usage_ctx, "_review_custody_lost", True)
        current_by_slot = {
            slot_id: rows[0]
            for slot_id, rows in current_groups.items()
            if slot_id not in ambiguous_slots
        }
        merged: List[Dict[str, Any]] = []
        for slot_id, original in originals.items():
            if str(original.get("operation_state") or "") == "custody_lost":
                merged.append(copy.deepcopy(original))
                setattr(usage_ctx, "_review_custody_lost", True)
                continue
            if slot_id in ambiguous_slots:
                merged.append(_custody_lost_row(
                    surface, slot_id, "current review roster contains duplicate slot ids", original,
                ))
                setattr(usage_ctx, "_review_custody_lost", True)
                continue
            if not _row_is_pending(original):
                merged.append(copy.deepcopy(original))
                continue
            candidate = current_by_slot.get(slot_id)
            operation_id = str(original.get("operation_id") or "")
            if candidate is not None and operation_id and str(
                candidate.get("operation_id") or ""
            ) == operation_id:
                merged.append(copy.deepcopy(candidate))
                continue
            lost = copy.deepcopy(original)
            lost["operation_state"] = "custody_lost"
            lost["late_result_pending"] = True
            merged.append(lost)
            setattr(usage_ctx, "_review_custody_lost", True)
        for slot_id in sorted(current_by_slot):
            row = current_by_slot[slot_id]
            if slot_id not in originals:
                extra = copy.deepcopy(row)
                extra["operation_state"] = "custody_lost"
                extra["late_result_pending"] = True
                merged.append(extra)
                setattr(usage_ctx, "_review_custody_lost", True)
        for slot_id in sorted(ambiguous_slots - set(originals)):
            rows = current_groups[slot_id]
            row = min(
                rows,
                key=lambda item: json.dumps(
                    item, sort_keys=True, ensure_ascii=True, default=str,
                ),
            )
            merged.append(_custody_lost_row(
                surface, slot_id, "current review roster contains duplicate slot ids", row,
            ))
        for row in sorted(
            invalid_rows,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True, default=str),
        ):
            merged.append(copy.deepcopy(row))
        return merged

    triad_now = getattr(usage_ctx, "_last_triad_raw_results", None)
    triad_rows = (
        list(triad_now) if isinstance(triad_now, (list, tuple))
        else [] if triad_now is None else [triad_now]
    )
    triad = _merge(
        "multi_model_review",
        triad_rows,
    )
    scope_now = getattr(usage_ctx, "_last_scope_raw_result", None)
    scope_wrapper = copy.deepcopy(scope_now) if isinstance(scope_now, dict) else {}
    if isinstance(scope_now, dict):
        raw_rows = scope_now.get("raw_results") if "raw_results" in scope_now else None
        scope_rows = (
            list(raw_rows) if isinstance(raw_rows, (list, tuple))
            else [raw_rows] if raw_rows is not None
            else [scope_now] if scope_now else []
        )
    else:
        scope_rows = [] if scope_now is None else [scope_now]
    merged_scope = _merge("scope_review", scope_rows)
    if not scope_wrapper:
        frozen_wrapper = getattr(usage_ctx, "_review_frozen_scope_raw", None)
        scope_wrapper = copy.deepcopy(frozen_wrapper) if isinstance(frozen_wrapper, dict) else {}
    if merged_scope:
        scope_wrapper["raw_results"] = merged_scope
    usage_ctx._last_triad_raw_results = triad
    usage_ctx._last_scope_raw_result = scope_wrapper


def review_retry_cancelled(usage_ctx: Any) -> bool:
    """Return whether durable cancellation forbids another review send.

    Retry rails have more than one entry point (transport exceptions and
    format-repair/empty responses). Keep the durable stop check in one seam so
    every path observes the same fail-closed owner decision before dispatch.
    """
    if usage_ctx is not None:
        task_id = str(getattr(usage_ctx, "task_id", "") or "")
        drive_root = getattr(usage_ctx, "drive_root", None)
        if task_id and drive_root:
            try:
                from ouroboros.cancel_intents import (
                    _validated_retry_root_cancel_key,
                    cancel_pending,
                )

                if cancel_pending(drive_root, task_id, strict=True):
                    return True
                retry_root = _validated_retry_root_cancel_key(drive_root, task_id)
                if retry_root and cancel_pending(drive_root, retry_root, strict=True):
                    return True
            except Exception:
                # An unreadable cancel projection must not authorize another
                # paid/retryable physical attempt.
                return True
    return False


def retryable_review_exception(
    exc: Exception, usage_ctx: Any,
    attempt_history: Optional[_ReviewAttemptHistory] = None,
) -> bool:
    """Whether a second byte-identical review send has a terminal basis."""
    from ouroboros.loop_llm_call import classify_llm_exception

    def _annotate_unknown_outcome() -> None:
        # Provider exception types may expose ``code`` as a read-only property.
        # This annotation is diagnostic only; it must never replace the
        # original failure or weaken the no-resend decision.
        try:
            if not str(getattr(exc, "code", "") or ""):
                setattr(exc, "code", "provider_outcome_unknown")
        except Exception:
            pass

    # A cancellation requested after the first route attempt is a terminal
    # owner decision for this wave, even when the transport wrapper surfaced a
    # generic retryable preparation error. Reusing the existing durable cancel
    # intent prevents a zero-cost pre-dispatch retry from racing the stop rail.
    if review_retry_cancelled(usage_ctx):
        return False

    history = attempt_history
    if history is None:
        history = _ReviewAttemptHistory()
        history.observe(exc)
    if history.unknown_outcome_seen:
        if usage_ctx is not None:
            setattr(usage_ctx, "_review_custody_lost", True)
        _annotate_unknown_outcome()
        return False

    classification = classify_llm_exception(exc)
    if classification.kind == "provider_outcome_unknown":
        if usage_ctx is not None:
            setattr(usage_ctx, "_review_custody_lost", True)
        _annotate_unknown_outcome()
    return classification.retry_same_request


def _attempt_key(request: Any, slot: Any) -> str:
    retry_key = str(getattr(request, "retry_key", "") or "").strip()
    slot_messages = (getattr(request, "slot_messages", {}) or {}).get(
        str(getattr(slot, "slot_id", "") or "")
    )
    slot_data = {
        key: getattr(slot, key, None) for key in (
            "slot_id", "model", "effort", "max_tokens", "temperature", "role_hint",
            "use_local", "route", "session_target", "session_profile",
            # Actor binding is operation identity: a changed configured-subagent
            # reference mints a new attempt key, exactly like a changed route.
            "subagent_id",
        )
    }
    identity = {
        "retry_key": retry_key,
        "surface": getattr(request, "surface", ""),
        "task_id": getattr(request, "task_id", ""),
        "call_type": getattr(request, "call_type", ""),
        "session_root": getattr(request, "session_root", ""),
        "slot": slot_data,
    }
    # An explicit retry key is the logical-operation identity. Mutable prompt
    # history must not turn a retry of that operation into a second paid send.
    # Callers change the key for genuinely new material/cycles; old callers
    # without one retain the full content-addressed behavior byte-for-byte.
    if retry_key:
        payload = identity
    else:
        payload = {
            **identity,
            "task_attempt": getattr(request, "task_attempt", None),
            "max_tokens": getattr(request, "max_tokens", None),
            "temperature": getattr(request, "temperature", None),
            "no_proxy": getattr(request, "no_proxy", None),
            "messages": slot_messages if slot_messages is not None else getattr(request, "messages", []),
            "goal": getattr(request, "goal", ""),
            "scope": getattr(request, "scope", ""),
            "subject": getattr(request, "subject", ""),
            "evidence_refs": getattr(request, "evidence_refs", []),
            "evidence": getattr(request, "evidence", {}),
            "policy": getattr(request, "policy", {}),
            "session_task": getattr(request, "session_task", ""),
        }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _logical_timeout(slot: Any, request: Any, usage_meta: Dict[str, Any]) -> float:
    deadline = str(getattr(request, "deadline_at", "") or "").strip()
    if not deadline and isinstance(usage_meta, dict):
        deadline = str(usage_meta.get("deadline_at") or "").strip()
    from ouroboros.config import get_finalization_grace_sec

    return review_operation_timeout_sec(
        getattr(slot, "timeout_sec", None),
        route=getattr(slot, "route", None),
        deadline_at=deadline,
        transport_timeout_sec=getattr(slot, "transport_timeout_sec", None),
        reserve_sec=get_finalization_grace_sec(),
    )


def _emit_operation(
    usage_ctx: Any,
    *,
    task_id: str,
    request: Any,
    entry: ActiveReviewAttempt,
    slot: Any,
    phase: str,
    **extra: Any,
) -> None:
    if usage_ctx is None:
        return
    try:
        event_queue = getattr(usage_ctx, "event_queue", None)
        values = {
            "task_id": task_id,
            "operation_id": entry.operation_id,
            "phase": phase,
            "kind": "review",
            "task_attempt": getattr(request, "task_attempt", None),
            "execution_id": str(getattr(usage_ctx, "execution_id", "") or ""),
            # ReviewRequest has no separate round carrier; the unique physical
            # operation is the stable round identity for both terminal events.
            "round_id": str(getattr(request, "round_id", "") or entry.operation_id),
            "slot_id": str(getattr(slot, "slot_id", "") or ""),
            **extra,
        }
        if event_queue is not None:
            emit_cognitive_operation_event(event_queue, **values)
            return
        from ouroboros.tools.review_helpers import emit_review_event

        emit_review_event(usage_ctx, {"type": "cognitive_operation", **values})
    except Exception:
        log.debug("review operation event failed", exc_info=True)


def _late_or_timeout_actor(
    slot: Any, entry: Any, timeout: float, error_actor: Callable[..., Any],
) -> Any:
    if entry is not None:
        with _ACTIVE_LOCK:
            if entry.event.is_set() and entry.actor is not None:
                try:
                    return copy.deepcopy(entry.actor)
                except Exception:
                    return entry.actor
            entry.timed_out = True
    actor = error_actor(
        slot,
        f"Timeout after {timeout:g}s; physical review operation remains in flight",
        entry.operation_id if entry is not None else "",
        "in_flight" if entry is not None else "settled",
    )
    if entry is not None and entry.retry_state:
        usage = dict(getattr(actor, "usage", None) or {})
        usage.update({
            key: str(entry.retry_state.get(key) or "")
            for key in ("pending_invocation_id", "delegated_run_id")
            if entry.retry_state.get(key)
        })
        actor.usage = usage
    return actor


def _settle_review_attempt(
    entry: ActiveReviewAttempt,
    slot: Any,
    actor: Any,
    *,
    usage_ctx: Any,
    request: Any,
    task_id: str,
    result_queue: "queue.Queue[Any]",
) -> None:
    """Publish one physical review settlement to process-local custody."""
    with _ACTIVE_LOCK:
        late = bool(entry.timed_out)
        failure_custody = getattr(actor, "usage", None) or {}
        physical_attempt_state = str(
            failure_custody.get("physical_attempt_state") or ""
        ).strip().lower()
        malformed_physical_state = bool(
            physical_attempt_state
            and physical_attempt_state not in PHYSICAL_ATTEMPT_STATES
        )
        if malformed_physical_state:
            # Treat malformed provenance as an unknown paid outcome.  A typed
            # HTTP status cannot make an unrecognized state safe to replay.
            physical_attempt_state = "unresolved"
            failure_custody["physical_attempt_state"] = physical_attempt_state
            failure_custody.pop("provider_status_code", None)
            if hasattr(actor, "usage"):
                actor.usage = failure_custody
        pending_invocation = str(failure_custody.get("pending_invocation_id") or "")
        terminal_provider_status = _terminal_provider_status(actor, failure_custody)
        if malformed_physical_state:
            terminal_provider_status = None
        capture_outcome_unknown = (
            physical_attempt_state in {"dispatched", "unresolved"}
            and terminal_provider_status is None
        )
        legacy_unknown = str(getattr(actor, "failure_code", "") or "") == "provider_outcome_unknown"
        # A typed terminal provider response is stronger than the legacy
        # catch-all code. The latter only means that custody is lost when no
        # terminal status survived the boundary.
        explicit_custody_lost = str(
            getattr(actor, "operation_state", "") or ""
        ).strip().lower() == "custody_lost"
        custody_lost = (
            explicit_custody_lost
            or malformed_physical_state
            or (legacy_unknown and terminal_provider_status is None)
            or capture_outcome_unknown
        )
        if (malformed_physical_state or capture_outcome_unknown) and str(getattr(actor, "failure_code", "") or "") != "provider_outcome_unknown":
            # The physical capture is stronger than a legacy/mocked actor code:
            # without a terminal provider response, a second paid send is unsafe.
            actor.failure_code = "provider_outcome_unknown"
        if malformed_physical_state and hasattr(actor, "http_status"):
            actor.http_status = None
        positive_physical_custody = physical_attempt_state in _POSITIVE_CAPTURE_STATES
        synthetic_not_dispatched = (
            str(getattr(actor, "operation_state", "") or "").strip().lower()
            == "not_dispatched"
            or str(getattr(actor, "status", "") or "").strip().lower()
            == "not_dispatched"
        )
        not_dispatched = synthetic_not_dispatched and not positive_physical_custody
        if positive_physical_custody and str(
            getattr(actor, "status", "") or ""
        ).strip().lower() == "not_dispatched":
            actor.status = "error"
            if not str(getattr(actor, "error", "") or ""):
                actor.error = (
                    "Positive physical-attempt custody contradicted a synthetic "
                    "not-dispatched status"
                )
        actor.operation_id = entry.operation_id
        actor.operation_state = (
            "custody_lost" if custody_lost else
            "not_dispatched" if not_dispatched else
            "in_flight" if pending_invocation else "late_settled" if late else "settled"
        )
        actor.late_result_pending = bool(pending_invocation or custody_lost)
        entry.actor = actor
        entry.event.set()
        if _ACTIVE.get(entry.key) is entry:
            if custody_lost:
                _NO_RESEND.setdefault(entry.key, entry.operation_id)
            _ACTIVE.pop(entry.key, None)
        # API transport errors may retry after settlement. A delegated session
        # has already spent its one run even when it settled as an error.
        pending = getattr(usage_ctx, "_review_pending_invocations", None)
        if pending_invocation and usage_ctx is not None:
            if not isinstance(pending, dict):
                pending = {}
                setattr(usage_ctx, "_review_pending_invocations", pending)
            pending[entry.key] = {
                "pending_invocation_id": pending_invocation,
                "operation_id": entry.operation_id,
            }
        if (
            bool(getattr(request, "reconcile_only", False))
            and str(getattr(actor, "failure_code", "") or "")
            == "review_custody_lost"
            and usage_ctx is not None
        ):
            setattr(usage_ctx, "_review_custody_lost", True)
        explicit_retry = bool(str(getattr(request, "retry_key", "") or "").strip())
        route = getattr(slot, "route", "")
        route_value = str(getattr(route, "value", route) or "")
        # A keyed plan/commit cycle owns one exact paid API attempt. Retain a
        # terminal API actor while the sibling cycle is still settling; otherwise
        # a retry can buy this already-settled slot again before the wave closes.
        keyed_terminal_api_error = bool(
            explicit_retry
            and (
                str(getattr(request, "surface", "") or "") == "plan_review"
                or str(getattr(request, "retry_key", "") or "").startswith(
                    "commit_review:"
                )
            )
            and route_value != "agent_session"
            and not pending_invocation
            # The actor's terminal state is the canonical settlement fact;
            # adapters are not required to populate the optional physical
            # capture metadata just to make same-cycle custody replayable.
            and actor.status == "error"
            and str(getattr(actor, "operation_state", "") or "")
            not in {"not_dispatched", "in_flight", "custody_lost"}
            # Explicit reserved/released states are pre-dispatch and stay
            # eligible for a real retry. A dispatched/unresolved state is
            # replayable only with a typed terminal HTTP response; otherwise
            # the capture-outcome guard above records custody_lost/no-resend.
            and physical_attempt_state in _PHYSICAL_CAPTURE_STATES
            and (
                physical_attempt_state in {"", "settled"}
                or terminal_provider_status is not None
            )
        )
        replayable = (
            actor.status in {"ok", "empty"}
            or keyed_terminal_api_error
            or bool(failure_custody.get("delegated_run_started") and not pending_invocation)
        )
        if replayable and usage_ctx is not None and (late or explicit_retry):
            settled = getattr(usage_ctx, "_review_settled_attempts", None)
            if not isinstance(settled, dict):
                settled = {}
                setattr(usage_ctx, "_review_settled_attempts", settled)
            try:
                settled[entry.key] = copy.deepcopy(actor)
            except Exception:
                log.debug("late review actor copy failed", exc_info=True)
    if not pending_invocation and not custody_lost:
        _emit_operation(
            usage_ctx, task_id=task_id, request=request, entry=entry, slot=slot,
            phase="failed" if actor.status == "error" else "finished",
        )
    if late and not pending_invocation and not custody_lost and usage_ctx is not None:
        try:
            from ouroboros.tools.review_helpers import emit_review_event

            emit_review_event(usage_ctx, {
                "type": "review_late_result",
                "task_id": task_id,
                "surface": getattr(request, "surface", ""),
                "slot_id": getattr(slot, "slot_id", ""),
                "operation_id": entry.operation_id,
                "status": actor.status,
                "response_ref": actor.response_ref,
                "operation_state": actor.operation_state,
            })
        except Exception:
            log.debug("late review result event failed", exc_info=True)
    result_queue.put(actor)


def run_custodied_review_slots(
    *,
    request: Any,
    slots: List[Any],
    usage_ctx: Any,
    task_id: str,
    usage_meta: Dict[str, Any],
    review_usage_scope: Any,
    run_slot: Callable[[Any, str, Dict[str, Any], float, Any], Any],
    error_actor: Callable[..., Any],
) -> List[Any]:
    """Run slots with independent logical windows and late-result custody."""
    from ouroboros.usage_accounting import usage_scope

    result_queue: "queue.Queue[Any]" = queue.Queue()
    slot_entries: Dict[str, ActiveReviewAttempt] = {}
    slot_deadlines: Dict[str, float] = {}
    slot_windows: Dict[str, float] = {}
    immediate_actors: Dict[str, Any] = {}
    paid_stamped = False

    def start(slot: Any) -> None:
        nonlocal paid_stamped
        key = _attempt_key(request, slot)
        slot_id = str(getattr(slot, "slot_id", "") or "")
        route = getattr(slot, "route", "")
        route_value = str(getattr(route, "value", route) or "")
        reserved_surface = (getattr(usage_ctx, "_review_reserved_operations", {}) or {}).get(
            str(getattr(request, "surface", "") or ""), {}
        )
        reserved_operation_id = str(reserved_surface.get(slot_id) or "") if isinstance(
            reserved_surface, dict
        ) else ""
        window = _logical_timeout(slot, request, usage_meta)
        slot_windows[slot_id] = window
        slot_deadlines[slot_id] = time.monotonic() + window
        custody_lost = False
        no_resend_operation_id = ""
        with _ACTIVE_LOCK:
            frozen_surfaces = getattr(usage_ctx, "_review_frozen_rows", None)
            frozen_surface = (
                frozen_surfaces.get(str(getattr(request, "surface", "") or ""), {})
                if isinstance(frozen_surfaces, dict) else {}
            )
            frozen_row = frozen_surface.get(slot_id) if isinstance(frozen_surface, dict) else None
            pending_attempts = getattr(usage_ctx, "_review_pending_invocations", None)
            retry_state = dict(
                pending_attempts.get(key, {}) if isinstance(pending_attempts, dict) else {}
            )
            settled = getattr(usage_ctx, "_review_settled_attempts", None)
            cached_actor = settled.get(key) if isinstance(settled, dict) else None
            if bool(getattr(request, "reconcile_only", False)) and isinstance(frozen_row, dict):
                if _row_is_pending(frozen_row):
                    token = str(frozen_row.get("pending_invocation_id") or "")
                    if not retry_state and token:
                        retry_state = {
                            "pending_invocation_id": token,
                            "operation_id": str(frozen_row.get("operation_id") or ""),
                        }
                else:
                    cached_actor = _frozen_actor(frozen_row, slot)
            retry_token = str(retry_state.get("pending_invocation_id") or "")
            retry_operation_id = str(retry_state.get("operation_id") or "")
            # Only delegated sessions have a durable invocation token that can
            # rejoin work after process-local custody is gone. An API row with
            # such a token is malformed state, not permission for a fresh send.
            exact_recovery = bool(
                retry_token and retry_operation_id and route_value == "agent_session"
            )
            retry_payload = dict(retry_state)
            retry_payload.pop("operation_id", None)
            if cached_actor is not None:
                try:
                    cached_actor = copy.deepcopy(cached_actor)
                except Exception:
                    log.debug("cached review actor copy failed", exc_info=True)
            entry = _ACTIVE.get(key)
            no_resend_operation_id = str(_NO_RESEND.get(key) or "")
            owner = False
            if no_resend_operation_id:
                entry = None
            elif (
                entry is not None
                and retry_operation_id
                and entry.operation_id != retry_operation_id
            ):
                # A logical key is not enough to join a different physical
                # operation. Preserve the real live entry and fail this caller.
                custody_lost = True
                entry = None
            elif cached_actor is not None:
                entry = ActiveReviewAttempt(
                    key=key,
                    operation_id=str(cached_actor.operation_id or new_call_id("review_replay")),
                    actor=cached_actor,
                )
                entry.event.set()
            elif (
                entry is None
                and bool(getattr(request, "reconcile_only", False))
                and not exact_recovery
            ):
                custody_lost = True
            elif entry is None and retry_state and not exact_recovery:
                custody_lost = True
            elif entry is None:
                if exact_recovery:
                    # Recovery is settlement, not fresh cognition. It gets one
                    # small structural join window even after the owner window
                    # is spent, and it reuses the already-paid operation id.
                    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC

                    window = float(NESTED_SETTLEMENT_MARGIN_SEC)
                    slot_windows[slot_id] = window
                    slot_deadlines[slot_id] = time.monotonic() + window
                if window > 0:
                    entry = ActiveReviewAttempt(
                        key=key,
                        operation_id=retry_operation_id or reserved_operation_id or new_call_id(
                            f"review_{getattr(request, 'surface', 'review')}_{getattr(slot, 'slot_id', 'slot')}"),
                        retry_state=retry_payload,
                    )
                    checkpoint = getattr(
                        usage_ctx, "_review_pending_invocation_checkpoint", None,
                    )
                    if callable(checkpoint):
                        surface = str(getattr(request, "surface", "") or "")
                        operation_id = entry.operation_id

                        def _checkpoint(
                            invocation_id: str,
                            *,
                            _surface: str = surface,
                            _slot_id: str = slot_id,
                            _operation_id: str = operation_id,
                            _callback: Callable[..., Any] = checkpoint,
                        ) -> None:
                            _callback(
                                surface=_surface,
                                slot_id=_slot_id,
                                operation_id=_operation_id,
                                invocation_id=invocation_id,
                            )

                        entry.pending_invocation_checkpoint = _checkpoint
                    _ACTIVE[key] = entry
                    owner = True
                    if isinstance(pending_attempts, dict):
                        pending_attempts.pop(key, None)
        if no_resend_operation_id:
            if usage_ctx is not None:
                setattr(usage_ctx, "_review_custody_lost", True)
            actor = error_actor(
                slot,
                "Provider outcome remains unknown; refusing a second physical review dispatch",
                no_resend_operation_id,
                "custody_lost",
            )
            actor.failure_code = "provider_outcome_unknown"
            actor.late_result_pending = True
            immediate_actors[slot_id] = actor
            return
        if custody_lost:
            if usage_ctx is not None:
                setattr(usage_ctx, "_review_custody_lost", True)
            immediate_actors[slot_id] = error_actor(
                slot,
                "Exact review custody is unavailable; refusing a second paid dispatch",
                "",
                "custody_lost",
            )
            return
        if entry is None:
            immediate_actors[slot_id] = error_actor(
                slot, "Owner deadline exhausted before physical review dispatch",
                reserved_operation_id, "not_dispatched",
            )
            return
        slot_entries[slot_id] = entry
        if window <= 0 and cached_actor is None:
            immediate_actors[slot_id] = _late_or_timeout_actor(
                slot, entry, window, error_actor,
            )
            return
        if owner:
            if not paid_stamped and not retry_state:
                from ouroboros.review_dispatch import stamp_review_paid_on_dispatch

                try:
                    stamp_review_paid_on_dispatch(usage_ctx)
                except Exception:
                    with _ACTIVE_LOCK:
                        if _ACTIVE.get(entry.key) is entry:
                            _ACTIVE.pop(entry.key, None)
                    slot_entries.pop(slot_id, None)
                    raise
                paid_stamped = True
            def worker() -> None:
                _emit_operation(
                    usage_ctx, task_id=task_id, request=request, entry=entry, slot=slot,
                    phase="started",
                )
                try:
                    with usage_scope(review_usage_scope):
                        actor = run_slot(
                            slot, entry.operation_id, entry.retry_state,
                            slot_deadlines[slot_id],
                            entry.pending_invocation_checkpoint,
                        )
                except Exception as exc:
                    operation_state = _worker_exception_operation_state(
                        exc, entry.retry_state,
                    )
                    actor = error_actor(
                        slot, f"{type(exc).__name__}: {exc}", entry.operation_id,
                        operation_state,
                    )
                    _attach_worker_exception_facts(actor, exc, entry.retry_state)
                _settle_review_attempt(
                    entry, slot, actor, usage_ctx=usage_ctx, request=request,
                    task_id=task_id, result_queue=result_queue,
                )

            threading.Thread(
                target=worker,
                name=f"ouroboros-review-{getattr(request, 'surface', 'review')}-{slot_id}",
                daemon=True,
            ).start()
        elif cached_actor is not None:
            result_queue.put(cached_actor)
        else:
            def relay() -> None:
                entry.event.wait()
                if entry.actor is not None:
                    try:
                        result_queue.put(copy.deepcopy(entry.actor))
                    except Exception:
                        result_queue.put(entry.actor)

            threading.Thread(
                target=relay,
                name=f"ouroboros-review-relay-{slot_id}",
                daemon=True,
            ).start()

    for slot in slots:
        start(slot)

    actors: List[Any] = list(immediate_actors.values())
    pending = {
        str(getattr(slot, "slot_id", "") or "") for slot in slots
        if str(getattr(slot, "slot_id", "") or "") not in immediate_actors
    }
    while pending:
        now = time.monotonic()
        expired = {slot_id for slot_id in pending if slot_deadlines[slot_id] <= now}
        for slot_id in expired:
            pending.remove(slot_id)
        if not pending:
            break
        remaining = min(slot_deadlines[slot_id] - now for slot_id in pending)
        try:
            actor = result_queue.get(timeout=remaining)
        except queue.Empty:
            continue
        if actor.slot_id in pending:
            actors.append(actor)
            pending.remove(actor.slot_id)

    returned_ids = {str(getattr(actor, "slot_id", "") or "") for actor in actors}
    for slot in slots:
        slot_id = str(getattr(slot, "slot_id", "") or "")
        if slot_id in returned_ids:
            continue
        entry = slot_entries.get(slot_id)
        timeout = slot_windows.get(slot_id)
        if timeout is None:
            timeout = _logical_timeout(slot, request, usage_meta)
        actors.append(_late_or_timeout_actor(slot, entry, timeout, error_actor))
    return actors


def review_retry_custody_available(
    *,
    retry_key: str,
    surface: str,
    task_id: str,
    call_type: str,
    session_root: str,
    slots: List[Any],
    usage_ctx: Any,
) -> bool:
    """Whether a same-cycle retry can join/replay without a new dispatch.

    This is intentionally process-local, matching the custody store itself. A
    caller that recovered a durable ``in_flight`` wave after process loss must
    report the outcome as unknown instead of turning absence of local custody
    into permission for a second paid send.
    """
    request = SimpleNamespace(
        retry_key=str(retry_key or ""),
        surface=str(surface or ""),
        task_id=str(task_id or ""),
        call_type=str(call_type or ""),
        session_root=str(session_root or ""),
    )
    settled = getattr(usage_ctx, "_review_settled_attempts", None)
    pending = getattr(usage_ctx, "_review_pending_invocations", None)
    with _ACTIVE_LOCK:
        for slot in slots:
            key = _attempt_key(request, slot)
            if key in _ACTIVE:
                continue
            if key in _NO_RESEND:
                continue
            if isinstance(settled, dict) and key in settled:
                continue
            if isinstance(pending, dict) and key in pending:
                continue
            return False
    return True
