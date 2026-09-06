"""
LLM call, retry, pricing, and usage-event logic for the main loop.

Handles model pricing estimation, cost tracking, per-call retry with backoff,
and real-time usage event emission.
Extracted from loop.py to keep the main loop orchestrator focused.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros import model_concurrency
from ouroboros.anthropic_native_custody import public_custody_projection
from ouroboros.config import get_finalization_grace_sec  # noqa: F401 - legacy monkeypatch seam
from ouroboros.deadline_utils import (
    owner_deadline_exhausted,
    seconds_until,
    transport_timeout_with_deadline,
)
from ouroboros.llm import LLMClient, LocalContextTooLargeError, add_usage
from ouroboros.llm_attempt import PROVIDER_POLICY_REFUSAL, _is_provider_policy_refusal  # typed-refusal contract owner
from ouroboros.observability import new_call_id, new_execution_id, persist_call
from ouroboros.pricing import emit_llm_usage_event, estimate_cost_optional, infer_model_category
from ouroboros.provider_models import provider_for_model
from ouroboros.transport_custody import attempt_custody_event_fields, is_pre_dispatch_transport_failure, is_retryable_transport_death
from ouroboros._usage_response import provider_cost_value as _provider_cost_value
from ouroboros.usage_accounting import (
    PhysicalAttemptContext,
    UsageAccountingError,
    bind_physical_attempt_context,
)
from ouroboros.utils import (
    append_jsonl,
    emit_cognitive_operation_event,
    emit_main_llm_call_state_event,
    emit_log_event,
    sanitize_tool_result_for_log,
    truncate_review_artifact,
    utc_now_iso,
)

log = logging.getLogger(__name__)

MAIN_LOOP_MAX_TOKENS = 65_536


def _main_transport_timeout(
    model: str,
    deadline_ts: Optional[float],
    *,
    reserve_sec: Optional[float] = None,
) -> float:
    # Preserve the native Anthropic default while narrowing every route to an
    # owner deadline. Other routes use the shared dead-socket bound; local models
    # receive the same explicit bound their client already supports.
    explicit = 120 if provider_for_model(model) == "anthropic" else None
    # ``None`` is the low-level raw-deadline contract.  The production round
    # dispatcher passes the finalization reserve explicitly; keeping this
    # default raw prevents admission from accepting a call and then shrinking
    # its transport to the 0.001-second floor unexpectedly.
    return transport_timeout_with_deadline(
        explicit, deadline_ts=deadline_ts,
        reserve_sec=0.0 if reserve_sec is None else reserve_sec,
    )

# Retrieval transparency (v6.78.0, owner Q20/Q22): native provider web search happens
# INSIDE the solve model's own request, so `usage["web_search_sources"]` /
# `usage["server_tool_use"]` are the only host-attested evidence that the answer was
# grounded in fetched pages. `add_usage` accumulates numeric token keys only, so without
# this fold the fact dies at the per-call boundary and the acceptance reviewer never
# learns retrieval-vs-own-knowledge. Counts plus capped URLs only — no titles/snippets.
_RETRIEVAL_URL_CAP = 20
_RETRIEVAL_URL_CHARS = 200


def fold_retrieval_usage(accumulated_usage: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate ONE call's native-retrieval facts onto the running usage dict.

    Bounding here is DISCLOSED, never silent (BIBLE P1), on the same three-part contract
    as ``_outcome_receipts.disclosed_list_projection``: bounded values, an exact
    ``urls_omitted`` count, and ``urls_identity_sha256`` over the FULL set — which the
    reviewer-facing projection in ``review_evidence`` then re-uses through that shared
    helper. This function cannot BE that helper: it is a streaming accumulator folded
    once per LLM call over a whole task, so it never holds the complete list the
    one-shot projection takes as input. What it can share, and does, is the per-string
    bound (the SSOT ``truncate_review_artifact``, so a clipped URL carries its own
    omission note instead of being silently shortened) and an O(1)-memory rolling chain
    hash standing in for the full-set hash."""
    sources = usage.get("web_search_sources")
    sources = [s for s in sources if isinstance(s, dict)] if isinstance(sources, list) else []
    server_tool_use = usage.get("server_tool_use") if isinstance(usage.get("server_tool_use"), dict) else {}
    try:
        requests = int(server_tool_use.get("web_search_requests") or 0)
    except (TypeError, ValueError):
        requests = 0
    if not requests and sources:
        requests = 1  # provider reported citations without a request counter
    if not requests and not sources:
        return
    record = accumulated_usage.get("retrieval")
    record = record if isinstance(record, dict) else {"web_search_requests": 0, "source_count": 0, "urls": []}
    record["web_search_requests"] = int(record.get("web_search_requests") or 0) + requests
    # Total fetched sources, so a `urls` list clipped at the cap stays disclosed.
    record["source_count"] = int(record.get("source_count") or 0) + len(sources)
    urls = record.get("urls")
    urls = list(urls) if isinstance(urls, list) else []
    # Dedup keys over the FULL RAW url, one per RETAINED entry (stays O(cap) memory).
    # Round 3: deduping on the RENDERED value silently lost evidence — two DISTINCT
    # long URLs sharing the retained prefix and length render byte-identically, so the
    # second was skipped while `urls_omitted` reported 0, i.e. a fetched URL vanished
    # from evidence promising an exact omission count (BIBLE P1). Raw keys make a
    # repeat a true repeat and every distinct URL either carried or counted.
    seen = record.get("urls_dedup_sha256")
    seen = list(seen) if isinstance(seen, list) else []
    omitted = int(record.get("urls_omitted") or 0)
    identity = str(record.get("urls_identity_sha256") or "")
    for source in sources:
        raw = str(source.get("url") or "").strip()
        if not raw:
            continue
        # Rolling chain hash over every fetched URL's RAW value (never the rendered one,
        # which two distinct URLs can share) in arrival order: an O(1)-memory
        # durable identity of the FULL set (recomputable from the per-call `llm_response`
        # observability payloads, which keep the raw `usage`), so the bounded list below
        # stays checkable against what the model actually fetched.
        identity = hashlib.sha256(f"{identity}\n{raw}".encode("utf-8")).hexdigest()
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if key in seen:
            continue  # the SAME url fetched again, already carried in `urls`
        if len(urls) < _RETRIEVAL_URL_CAP:
            urls.append(truncate_review_artifact(raw, limit=_RETRIEVAL_URL_CHARS))
            seen.append(key)
        else:
            omitted += 1
    record["urls"] = urls
    record["urls_dedup_sha256"] = seen
    # Exact count of fetched URLs the capped list does NOT carry (repeats among the
    # omitted are counted each time — the identity hash covers the exact full sequence).
    record["urls_omitted"] = omitted
    record["urls_identity_sha256"] = identity
    accumulated_usage["retrieval"] = record

# Per-class retry policy: TRANSIENT provider failures (finish_reason=null
# glitches, 429/5xx/overloaded) get a larger same-model attempt budget than
# permanent classes, because the owner deliberately runs single-model setups
# (all slots = one model, empty fallback) for clean measurement and a 3-attempt
# cap turned recoverable provider blips into whole-task "No viable fallback
# model configured" deaths. Permanent classes (auth/quota/bad_request/too
# large) keep failing fast. There is NO cross-model fallback here — the same
# request is retried on the SAME model.
_TRANSIENT_RETRY_KINDS = frozenset({"provider_transient", "provider_incomplete_response"})
# OB-01: stamped when THIS invocation spent its same-model retry wall without a
# usable response; entry-cleared per invocation; PERMANENT classes leave it unspent.
RETRY_WALL_EXHAUSTED_KEY = "_llm_retry_wall_exhausted"
# Error kinds that put a model on the F1 fallback cooldown. Superset of the same-model
# retry kinds: a body-error 429 (HTTP 200 with an error in the body — the canonical
# cloud.ru/OpenRouter rate-limit shape) is classified "rate_limit", which must cool the
# model down even though it is not a same-model retry kind. Kept separate so widening the
# cooldown trigger never enlarges the same-model transient-retry budget.
_COOLDOWN_ERROR_KINDS = _TRANSIENT_RETRY_KINDS | frozenset({"rate_limit"})
# A subscription window that is spent but heals on a timer. SSOT for the name.
# Deliberately NOT `quota_exhausted`: that class is classified PERMANENT, which is
# correct for a billing refusal (402, no credits) and wrong for a plan window whose
# only cure is waiting. Scheduling follows `reset_at`, not the 60s-capped exponential
# backoff, so a six-hour window never becomes sixty one-minute retries.
SUBSCRIPTION_WINDOW_EXHAUSTED = "subscription_window_exhausted"
_TRANSIENT_RETRY_DEFAULT = 6
_TRANSIENT_BACKOFF_CAP_SEC = 60.0
# Stop retrying when the remaining task deadline cannot absorb the backoff
# sleep plus a useful follow-up attempt — burning the last minutes of a task
# deadline sleeping between retries is worse than failing visibly.
_DEADLINE_RETRY_FLOOR_SEC = 10.0
# Bounded PAID repeat of a dispatched request whose socket died with a typed
# transport death (transport_custody.is_retryable_transport_death — never a
# timeout, status, body error or pre-dispatch failure): the primary main-loop
# round dispatch alone opts in, at most this many extra physical sends per ROUND
# (≤3 unresolved upper bounds), each its own ledger attempt; backoff by death
# ordinal. Every other caller keeps the default 0 = the no-resend doctrine.
_TRANSPORT_DEATH_RETRIES = 2
_TRANSPORT_DEATH_BACKOFF_SEC = (4.0, 8.0)
# Round-keyed repeat counter on the shared usage dict ({"round_id", "count"}):
# the wait episode's free redial re-enters call_llm_with_retry with the SAME
# round id, so the budget must not re-arm per invocation.
TRANSPORT_DEATHS_KEY = "_transport_deaths"


def transient_retry_max(default_retries: int) -> int:
    """Attempt budget for transient provider failure classes.

    Tunable via OUROBOROS_TRANSIENT_RETRY_MAX (SSOT default in
    config.SETTINGS_DEFAULTS); never below the caller's default budget so
    misconfiguration cannot reduce existing resilience.
    """
    try:
        from ouroboros.config import SETTINGS_DEFAULTS
        default_value = int(SETTINGS_DEFAULTS.get("OUROBOROS_TRANSIENT_RETRY_MAX", _TRANSIENT_RETRY_DEFAULT))
    except Exception:
        default_value = _TRANSIENT_RETRY_DEFAULT
    raw = os.environ.get("OUROBOROS_TRANSIENT_RETRY_MAX", "").strip()
    try:
        value = int(raw) if raw else default_value
    except ValueError:
        value = default_value
    return max(int(default_retries), value)


def _empty_response_log_msg(usage: Dict[str, Any], is_provider_glitch: bool) -> str:
    """Honest message for an empty/incomplete LLM response: a transient provider
    body-error (OpenRouter 429/5xx inside an HTTP 200, surfaced as usage
    ``provider_error``) that a same-model reroute could not escape is named as
    itself, not as a blank finish_reason=null glitch. Pure: the error KIND is owned
    by the caller's typed assignment (``_cooldown_kind_for_empty_response``)."""
    provider_error = usage.get("provider_error") if isinstance(usage, dict) else None
    if isinstance(provider_error, dict):
        return f"Provider returned a body error (code={provider_error.get('code')}): {provider_error.get('message')}"
    if is_provider_glitch:
        return "Provider returned incomplete response (finish_reason=null)"
    return "LLM returned empty response (no content, no tool_calls)"


def _classify_empty_response(usage: Dict[str, Any], msg: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Classify an empty / no-tool-call response → (event_type, is_provider_glitch,
    permanent_body_error). A TYPED non-transient body error (WA1 kind
    ``provider_error``: auth / quota / bad_request) is PERMANENT — a same-model
    reroute already failed in the transport, so retrying here only burns the
    transient budget. Only rate_limit / provider_transient body errors and a bare
    ``finish_reason=null`` glitch are retryable."""
    finish_reason = msg.get("finish_reason") or msg.get("stop_reason")
    if str(finish_reason or "").strip().lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
        return "remote_context_overflow", False, True
    body_err = usage.get("provider_error") if isinstance(usage, dict) else None
    if isinstance(body_err, dict) and any(
        str(body_err.get(key) or "").strip().lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES
        for key in ("code", "type")
    ):
        return "remote_context_overflow", False, True
    is_provider_glitch = finish_reason is None
    body_kind = str((body_err or {}).get("kind") or "") if isinstance(body_err, dict) else ""
    permanent_body_error = bool(body_err) and body_kind not in ("rate_limit", "provider_transient")
    if permanent_body_error:
        event_type = "provider_body_error"
    elif is_provider_glitch:
        event_type = "provider_incomplete_response"
    else:
        event_type = "llm_empty_response"
    return event_type, is_provider_glitch, permanent_body_error


def _attempt_loop_budget(max_retries: int, attempt_cap: Optional[int]) -> int:
    """Attempt-loop ceiling. Normally ``transient_retry_max(max_retries)``; when
    ``attempt_cap`` is set (F1 fallback candidate), cap the WHOLE loop (every error class)
    to a small total so the chain tries a candidate a fixed couple of times then moves on.
    Applied only to candidates; the primary passes None and keeps its full budgets."""
    budget = transient_retry_max(max_retries)
    if attempt_cap is not None:
        budget = max(1, min(int(budget), int(attempt_cap)))
    return budget


def _record_and_emit_empty_response(
    *, usage, msg, accumulated_usage, event_queue, drive_logs, task_id, execution_id,
    round_id, llm_call_id, round_idx, attempt, model, task_type, content, tool_calls,
    request_ref, response_ref, transient_budget, context_fit_event_fields,
    task_attempt=None,
) -> tuple:
    """Classify an empty / no-tool-call response, log + emit its events, and stamp
    accumulated_usage (last error / execution_status / reason_code / F1 cooldown kind).
    Returns ``(event_type, is_provider_glitch, permanent_body_error)`` for the caller's
    retry decision. Extracted from call_llm_with_retry to keep that loop readable."""
    finish_reason = msg.get("finish_reason") or msg.get("stop_reason")
    body_error = usage.get("provider_error") if isinstance(usage, dict) and isinstance(usage.get("provider_error"), dict) else {}
    event_type, is_provider_glitch, permanent_body_error = _classify_empty_response(usage, msg)
    log_msg = _empty_response_log_msg(usage, is_provider_glitch)
    log.warning("%s, attempt %d/%d", log_msg, attempt + 1, transient_budget)
    _emit_empty_response_events(
        event_type, event_queue=event_queue, drive_logs=drive_logs,
        base={"task_id": task_id, "execution_id": execution_id, "round_id": round_id,
              "llm_call_id": llm_call_id, "round": round_idx, "attempt": attempt + 1,
              "model": model, "finish_reason": finish_reason, "task_attempt": task_attempt,
              # WHICH upstream endpoint served (or refused) this round, plus the typed class
              # of an HTTP-200 body error (the shape issue #468 died on); a durable record
              # naming only the MODEL cannot attribute a same-model provider incident.
              "response_provider": usage.get("response_provider"),
              "provider_error_kind": body_error.get("kind"),
              **context_fit_event_fields},
        task_type=task_type,
        details={"content": content, "tool_calls": tool_calls,
                 "request_ref": request_ref, "response_ref": response_ref},
    )
    if event_type == "remote_context_overflow":
        status, reason, kind = "infra_failed", "llm_api_error", "context_overflow"
    else:
        status = "infra_failed" if (is_provider_glitch and not permanent_body_error) else "failed"
        reason = event_type
        # Cooldown signal for the F1 fallback gate (see helper; not a retry change).
        kind = _cooldown_kind_for_empty_response(body_error, event_type)
    accumulated_usage.update({
        "_last_llm_error": _short_error_text(log_msg), "execution_status": status,
        "reason_code": reason, "_last_llm_error_kind": kind,
    })
    return event_type, is_provider_glitch, permanent_body_error


def _cooldown_kind_for_empty_response(body_error: Dict[str, Any], event_type: str) -> str:
    """Pick the kind exposed as ``_last_llm_error_kind`` for the F1 fallback-chain cooldown
    gate on an empty/body-error response, from the caller's validated body error ({} when
    there is none). PREFER the provider body-error kind (a 429 surfaces as ``rate_limit``)
    so a rate-limited model cools down regardless of finish_reason; otherwise fall back to
    ``event_type`` (``provider_incomplete_response`` cools; ``provider_body_error`` /
    ``llm_empty_response`` are not in the cooldown set, so they correctly do not). Purely
    the cooldown SIGNAL — it does not change the same-model transient-retry layering (the
    primary keeps its full plan-preserved budget; cooldown is the second layer)."""
    body_kind = str(body_error.get("kind") or "")
    if body_error:
        body_code = str(body_error.get("code") or "").strip()
        body_message = str(body_error.get("message") or "")
        if body_code == "413" or _output_or_body_size_message(body_message):
            return "request_too_large"
    return body_kind if body_kind in _COOLDOWN_ERROR_KINDS else event_type


def _retry_backoff_sec(
    accumulated_usage: Dict[str, Any], error_kind: str, attempt: int, is_transient: bool,
) -> float:
    """Seconds to wait before retrying the same request.

    A KNOWN reset instant wins over guesswork: a spent subscription window is scheduled
    against its own ``reset_at``, never through the 60s-capped exponential, so a
    six-hour window never becomes sixty one-minute retries. ``_sleep_within_deadline``
    then honestly refuses when the task deadline cannot absorb that wait.
    """
    retry_after = accumulated_usage.get("_last_llm_retry_after_sec")
    if error_kind == SUBSCRIPTION_WINDOW_EXHAUSTED and retry_after is not None:
        return max(0.0, float(retry_after))
    return min(2.0 ** attempt * 4, _TRANSIENT_BACKOFF_CAP_SEC if is_transient else 30.0)


def _sleep_within_deadline(seconds: float, deadline_ts: Optional[float]) -> bool:
    """Sleep ``seconds`` if the task deadline (epoch seconds) allows another
    attempt afterwards. Returns False — without sleeping — when the remaining
    time budget cannot absorb the backoff, signalling the caller to stop."""
    if deadline_ts:
        remaining = float(deadline_ts) - time.time()
        if remaining < float(seconds) + _DEADLINE_RETRY_FLOOR_SEC:
            return False
    time.sleep(float(seconds))
    return True


def _emit_retry_deadline_exhausted(
    drive_logs: pathlib.Path,
    *,
    task_id: str,
    execution_id: str,
    round_id: str,
    round_idx: int,
    attempt: int,
    model: str,
    error_kind: str,
) -> None:
    """Durable record that retries stopped because the deadline could not
    absorb another backoff sleep (emitted by BOTH transient failure paths)."""
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "llm_retry_deadline_exhausted",
        "task_id": task_id,
        "execution_id": execution_id,
        "round_id": round_id,
        "round": round_idx, "attempt": attempt + 1,
        "model": model,
        "error_kind": error_kind,
    })


def _deadline_not_dispatched(
    deadline_ts: Optional[float], accumulated_usage: Dict[str, Any],
    drive_logs: pathlib.Path, *, task_id: str, model: str, round_idx: int,
    event_queue: Optional[queue.Queue] = None, llm_call_id: str = "",
    task_attempt: Any = None, execution_id: str = "", round_id: str = "",
    reserve_sec: Optional[float] = None,
) -> bool:
    # ``None`` preserves the low-level helper's raw-deadline contract.  The
    # production round dispatcher supplies the finalization reserve explicitly;
    # direct callers such as retry diagnostics can still exercise a first
    # admitted attempt and let the retry backoff gate stop the next one.
    if not owner_deadline_exhausted(deadline_ts=deadline_ts, reserve_sec=0.0 if reserve_sec is None else reserve_sec):
        return False
    if llm_call_id:
        _emit_llm_operation(
            event_queue, task_id, llm_call_id, "failed", task_attempt,
            execution_id, round_id,
        )
    # Record the refusal without minting an LLM operation or attempt. While THIS
    # round holds an unresolved attempt (its record) AND the sticky kind is unknown
    # (a pending granted repeat, or the unknown terminal) the kind must not become
    # `deadline_exhausted` — that would re-open the forced-final rail over a possibly
    # live request — and a never-sent grant is un-counted; a refused FREE redial
    # keeps the base stamps (the record still fences forced-final and the chain).
    if _transport_death_repeats(accumulated_usage, round_id) and accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown":
        if accumulated_usage.get("_last_llm_retry_same_request"):
            _uncount_transport_death(accumulated_usage)
    else:
        accumulated_usage.update(
            _last_llm_error_kind="deadline_exhausted",
            _last_llm_error="owner deadline exhausted before dispatch",
            _last_llm_retry_same_request=False,
            execution_status="infra_failed",
            reason_code="deadline_exhausted",
        )
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "llm_not_dispatched",
        "task_id": task_id,
        "round": round_idx,
        "model": model,
        "reason_code": "deadline_exhausted",
    })
    return True


@dataclass
class _LlmErrorContext:
    task_id: str
    task_type: str
    execution_id: str
    round_id: str
    llm_call_id: str
    round_idx: int
    attempt: int
    model: str
    request_ref: Optional[Dict[str, Any]]
    drive_logs: pathlib.Path
    event_queue: Optional[queue.Queue]
    accumulated_usage: Dict[str, Any]
    context_fit_event_fields: Optional[Dict[str, Any]] = None
    task_attempt: Any = None
    deadline_ts: Optional[float] = None
    max_retries: int = 0
    transient_budget: int = 0
    transport_death_retries: int = 0
    transport_reserve_sec: Optional[float] = None


@dataclass(frozen=True)
class LlmErrorClassification:
    kind: str
    retry_same_request: bool
    status_code: Optional[int] = None
    provider_code: str = ""
    # Wall-clock seconds until the failure can heal, when the provider states it.
    # Set only by classes whose recovery is a KNOWN INSTANT (a subscription window
    # reset), never by classes whose recovery is guesswork — those keep the ordinary
    # exponential backoff.
    retry_after_sec: Optional[float] = None
    reset_at: str = ""


def _emit_live_log(event_queue: Optional[queue.Queue], payload: Dict[str, Any]) -> None:
    """Thin wrapper around the SSOT helper — keeps the call-site signature stable."""
    emit_log_event(
        event_queue,
        {"ts": utc_now_iso(), **payload},
        log_label="LLM live",
    )


def _short_error_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


from ouroboros.context_budget import (  # one overflow vocabulary for every seam
    CONTEXT_OVERFLOW_CODES as _STRUCTURED_CONTEXT_OVERFLOW_CODES,
    context_overflow_message as _context_overflow_message,
    output_or_body_size_message as _output_or_body_size_message,
)
_FORCED_INCOMPLETE_FINISH_REASONS = _STRUCTURED_CONTEXT_OVERFLOW_CODES | frozenset({
    "length", "max_tokens", "tool_calls", "function_call", "tool_use",
})


def forced_response_is_incomplete(response_meta: Optional[Dict[str, Any]]) -> bool:
    """Return whether a forced provider response is not a complete final."""
    if not isinstance(response_meta, dict):
        return False
    finish_reason = response_meta.get("finish_reason")
    return bool(response_meta.get("tool_call_count")) or (
        response_meta.get("finish_reason_present") is True
        and (
            finish_reason is None
            or str(finish_reason).strip().lower() in _FORCED_INCOMPLETE_FINISH_REASONS
        )
    )


_NON_RETRYABLE_PROVIDER_MARKERS = {
    "quota_exhausted": (
        "insufficient credits",
        "insufficient_credit",
        "insufficient_quota",
        "quota exceeded",
        "billing",
        "payment required",
        "402",
    ),
    "auth_error": (
        "invalid_api_key",
        "unauthorized",
        "forbidden",
        "401",
        "403",
    ),
    "request_too_large": (
        "max_tokens",
        "maximum tokens",
        "output tokens",
        "maximum output",
        "too many tokens",
        "request body too large",
        "body too large",
    ),
    "bad_request": (
        "badrequest",
        "bad request",
        "conversation must end with a user message",
        "prefill",
        "unsupported",
        "invalid request",
        "400",
    ),
}
_RETRYABLE_PROVIDER_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "timeout",
    "temporarily",
    "server error",
    "502",
    "503",
    "504",
)
_RATE_LIMIT_TEXT_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "tokens per minute",
    "requests per minute",
    "token per minute",
    "request per minute",
    "tpm",
    "rpm",
)
_RETRYABLE_PROVIDER_CODES = frozenset({"rate_limit_exceeded"})


def _is_rate_limit_text(text: str) -> bool:
    low = str(text or "").lower()
    return any(marker in low for marker in _RATE_LIMIT_TEXT_MARKERS)


is_rate_limit_text = _is_rate_limit_text  # public SSOT aliases for the safety lane
NON_RETRYABLE_PROVIDER_MARKERS = _NON_RETRYABLE_PROVIDER_MARKERS


def _is_context_overflow_error(exc: Exception, safe_error: str) -> bool:
    """Classify untyped local/remote context-window overflow.

    The output-size precedence lives inside the SHARED helper, not here, so
    Main, the local transport, and the summarizer classify identically."""
    if isinstance(exc, LocalContextTooLargeError):
        return True
    provider_text = _exception_provider_message(exc, "")
    classification_text = "\n".join(
        value for value in (safe_error, provider_text) if str(value or "").strip()
    )
    if _is_rate_limit_text(classification_text):
        return False
    return _context_overflow_message(classification_text)


def _exception_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            try:
                return int(value)
            except ValueError:
                pass
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    capture = getattr(exc, "physical_attempt_capture", None)
    value = getattr(capture, "provider_status_code", None)
    return value if isinstance(value, int) else None


def _exception_body(exc: Exception) -> Dict[str, Any]:
    """Read a structured provider body without turning a diagnostic into a failure."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(exc, "response", None)
    parser = getattr(response, "json", None)
    if callable(parser):
        try:
            value = parser()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _exception_provider_values(exc: Exception) -> List[str]:
    values: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)

    for attr in ("code", "type"):
        add(getattr(exc, attr, None))
    body = _exception_body(exc)
    nested = body.get("error")
    for source in ((body, nested) if isinstance(nested, dict) else (body,)):
        for key in ("code", "type"):
            add(source.get(key))
    capture = getattr(exc, "physical_attempt_capture", None)
    if capture is not None:
        add(getattr(capture, "provider_code", None))
        add(getattr(capture, "provider_error_type", None))
    return values


def _exception_provider_code(exc: Exception, safe_error: str) -> str:
    del safe_error
    values = _exception_provider_values(exc)
    for value in values:
        if value.lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
            return value
    for value in values:
        if value.lower() in _RETRYABLE_PROVIDER_CODES:
            return value
    generic_bad_request_wrappers = {
        "badrequest",
        "badrequesterror",
        "invalidrequest",
        "invalidrequesterror",
        "error",
    }
    for value in values:
        normalized = value.lower().replace("_", "").replace("-", "").replace(" ", "")
        if (
            not value.isdigit()
            and normalized not in generic_bad_request_wrappers
            and _provider_code_kind(value)
        ):
            return value
    # A response body that repeats only the generic numeric 400 is not a
    # provider-specific code. Leave it empty so its message can classify the
    # actual context/output failure; attr-only numeric codes stay available.
    body = _exception_body(exc)
    nested = body.get("error")
    body_values = [
        text for source in ((body, nested) if isinstance(nested, dict) else (body,))
        for key in ("code", "type") if (text := str(source.get(key) or "").strip())
    ]
    if body_values and all(
        value == "400"
        or value.lower().replace("_", "").replace("-", "").replace(" ", "")
        in generic_bad_request_wrappers
        for value in body_values
    ):
        return ""
    return values[0] if values else ""


def _exception_provider_message(exc: Exception, safe_error: str = "") -> str:
    """Best-effort human-readable provider error BODY text.

    Strict OpenAI-compatible providers (cloud.ru Foundation Models, vLLM/SGLang)
    return a 400 whose BODY distinguishes otherwise-identical status codes — e.g. a
    cloud.ru content-filter ("guardrails") block vs an ``Extra inputs are not
    permitted`` reasoning_content echo. ``provider_code`` alone cannot tell them
    apart, so surface the body message (sanitized + truncated) into the durable
    event for the owner. Pure read of ``exc.body``/repr; never changes routing."""
    body = _exception_body(exc)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict) and str(nested.get("message") or "").strip():
            return sanitize_tool_result_for_log(str(nested.get("message")))[:600]
        if str(body.get("message") or "").strip():
            return sanitize_tool_result_for_log(str(body.get("message")))[:600]
    text = str(safe_error or "").strip()
    return sanitize_tool_result_for_log(text)[:600] if text else ""


def _provider_code_kind(provider_code: str) -> str:
    code = str(provider_code or "").strip().lower()
    if not code:
        return ""
    for kind, markers in _NON_RETRYABLE_PROVIDER_MARKERS.items():
        if code == kind:
            return kind
        for marker in markers:
            normalized = str(marker).lower()
            if code == normalized or (not normalized.isdigit() and normalized in code):
                return kind
    return ""


def classify_llm_exception(exc: Exception, safe_error: str = "") -> LlmErrorClassification:
    """Classify provider errors without changing model/request semantics."""

    safe = safe_error or sanitize_tool_result_for_log(repr(exc))
    if isinstance(exc, LocalContextTooLargeError):
        return LlmErrorClassification("context_overflow", False)
    # Structured fact, not a keyword scan (Bible P5): a transport that KNOWS its
    # window is spent carries the typed code plus the reset instant.
    if str(getattr(exc, "code", "") or "") == SUBSCRIPTION_WINDOW_EXHAUSTED:
        reset_at = str(getattr(exc, "reset_at", "") or "")
        return LlmErrorClassification(
            SUBSCRIPTION_WINDOW_EXHAUSTED, True, _exception_status_code(exc),
            "", seconds_until(reset_at), reset_at,
        )
    status_code = _exception_status_code(exc)
    provider_code = _exception_provider_code(exc, safe)
    # Typed refusal (llm_attempt.ProviderPolicyRefusal): nothing upstream answered,
    # permanent by class — structural, and it outranks every prose heuristic below.
    if _is_provider_policy_refusal(exc):
        return LlmErrorClassification(PROVIDER_POLICY_REFUSAL, False, status_code, provider_code or PROVIDER_POLICY_REFUSAL)
    provider_message = _exception_provider_message(exc, safe)
    classification_text = "\n".join(v for v in (safe, provider_message) if str(v or "").strip())
    low = classification_text.lower()
    if provider_code.lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
        return LlmErrorClassification("context_overflow", False, status_code, provider_code)
    provider_kind = _provider_code_kind(provider_code)
    # Named codes and numeric auth/quota codes are typed authority. Only a
    # numeric code that maps to generic bad_request defers to the semantic body
    # classifiers below, which can distinguish output/context failures.
    generic_numeric_bad_request = (
        provider_kind == "bad_request" and provider_code == "400"
    )
    if provider_kind and not generic_numeric_bad_request:
        return LlmErrorClassification(provider_kind, False, status_code, provider_code)
    if provider_code.lower() in _RETRYABLE_PROVIDER_CODES:
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if status_code == 429:
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if _is_rate_limit_text(low):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if _output_or_body_size_message(low):
        return LlmErrorClassification("request_too_large", False, status_code, provider_code)
    if _is_context_overflow_error(exc, classification_text):
        return LlmErrorClassification("context_overflow", False, status_code, provider_code)
    if provider_kind:
        return LlmErrorClassification(provider_kind, False, status_code, provider_code)
    for kind, markers in _NON_RETRYABLE_PROVIDER_MARKERS.items():
        if any(marker in low for marker in markers):
            return LlmErrorClassification(kind, False, status_code, provider_code)
    if status_code in {400, 401, 402, 403, 413, 422}:
        kind = {
            400: "bad_request",
            401: "auth_error",
            402: "quota_exhausted",
            403: "auth_error",
            413: "request_too_large",
            422: "bad_request",
        }[status_code]
        return LlmErrorClassification(kind, False, status_code, provider_code)
    if status_code == 408 or (status_code is not None and 500 <= status_code <= 599):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if status_code is not None and 400 <= status_code <= 499:
        return LlmErrorClassification("provider_error", False, status_code, provider_code)
    capture = getattr(exc, "physical_attempt_capture", None)
    if str(getattr(capture, "state", "") or "") in {"dispatched", "unresolved"}:
        # The provider may still finish a request whose socket outcome is
        # unknown. Neither a same-model retry nor a different paid fallback is
        # safe until a typed terminal provider fact exists.
        return LlmErrorClassification(
            "provider_outcome_unknown", False, status_code, provider_code,
        )
    if (
        str(getattr(capture, "state", "") or "") == "released"
        and str(getattr(capture, "provider", "") or "") != "local"
        and not bool(getattr(capture, "route_is_loopback", False))
        and is_pre_dispatch_transport_failure(exc)
    ):
        # Typed $0 fact: the request never left this host toward a REMOTE
        # provider (released custody + typed pre-dispatch transport failure).
        # Retrying the same request is free and safe; pacing/waiting is owned
        # by the round-level episode in loop.py, never by this helper. A local
        # provider's connect failure stays on the generic path — a stopped
        # local server is not a network outage worth waiting out — and so does
        # a loopback OpenAI-compatible route (Ollama / LM Studio / vLLM): its
        # provider stamp is remote-shaped but the server is on this host.
        return LlmErrorClassification(
            "transport_unavailable", True, status_code, provider_code,
        )
    if any(marker in low for marker in _RETRYABLE_PROVIDER_MARKERS):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    return LlmErrorClassification("provider_error", True, status_code, provider_code)


def _remember_llm_call(
    usage: Dict[str, Any],
    *,
    llm_call_id: str,
    execution_id: str,
    round_id: str,
    round_idx: int,
    attempt: int,
    model: str,
    display_model: str,
    provider: str,
    request_ref: Dict[str, Any],
    response_ref: Dict[str, Any],
) -> None:
    call_meta = {
        "llm_call_id": llm_call_id,
        "execution_id": execution_id,
        "round_id": round_id,
        "round": round_idx,
        "attempt": attempt,
        "model": model,
        "resolved_model": display_model,
        "provider": provider,
        "request_ref": request_ref.get("manifest_ref") if request_ref else None,
        "response_ref": response_ref.get("manifest_ref") if response_ref else None,
    }
    usage["_last_llm_call_meta"] = call_meta
    usage.setdefault("llm_call_refs", []).append(call_meta)


def _normalize_usage_cost(
    usage: Dict[str, Any],
    *,
    model: str,
    use_local: bool,
) -> tuple[Optional[float], str, str, bool]:
    raw_cost = usage.get("cost")
    provider_reported_cost = raw_cost is not None
    cost = _provider_cost_value(raw_cost) if provider_reported_cost else None
    display_model = str(usage.get("resolved_model") or model)
    provider = "local" if use_local else str(usage.get("provider") or "openrouter")
    if use_local:
        cost = 0.0
        display_model = f"{model} (local)"
    elif provider_reported_cost and cost is None:
        # MISSING falls through to the catalog estimate; a cost the provider DID
        # send but that cannot be trusted is honestly unknown RIGHT HERE. Shared
        # predicate: `_usage_response.provider_cost_value` — the lanes cannot fork.
        log.warning(
            "Provider reported an invalid cost (type=%s, value=%s) for %s; recording "
            "cost as unknown and skipping estimation",
            type(raw_cost).__name__,
            truncate_review_artifact(repr(raw_cost), limit=120),
            display_model,
        )
        usage["cost"] = None
        usage["cost_final"] = False  # unknown cost is never a closed book
        return None, display_model, provider, bool(usage.get("cost_estimated"))
    elif cost is None:
        cost = estimate_cost_optional(
            display_model,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            cache_usage={
                "cached_tokens": int(usage.get("cached_tokens") or 0),
                "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
                "cache_write_tokens_by_ttl": (
                    usage.get("cache_write_tokens_by_ttl")
                    if isinstance(usage.get("cache_write_tokens_by_ttl"), dict)
                    else None
                ),
            },
            provider=provider,
        )
    usage["cost"] = cost
    cost_estimated = bool(usage.get("cost_estimated")) or (cost is not None and not provider_reported_cost)
    return cost, display_model, provider, cost_estimated


def _uncount_transport_death(accumulated_usage: Dict[str, Any]) -> None:
    """A granted repeat that never left the host is no repeat: take it back off the round record."""
    record = accumulated_usage.get(TRANSPORT_DEATHS_KEY) or {}
    record["count"] = int(record.get("count") or 0) - 1
    if record["count"] <= 0:
        accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
    accumulated_usage["_last_llm_retry_same_request"] = False


def _transport_death_repeats(accumulated_usage: Dict[str, Any], round_id: str) -> int:
    """Paid repeats already spent on THIS round's transport deaths; a record from
    another round is dropped (and a usable response pops it) so neither the
    budget nor the terminal hint ever reads a stale count."""
    record = accumulated_usage.get(TRANSPORT_DEATHS_KEY)
    if isinstance(record, dict) and record.get("round_id") == round_id:
        return int(record.get("count") or 0)
    accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
    return 0


def _record_llm_call_error(
    error: Exception,
    ctx: _LlmErrorContext,
) -> bool:
    """Record and classify an LLM-round exception.

    Emits live/durable error evidence, marks usage as infra-failed, and returns
    whether the caller must stop retrying the unchanged request.
    """
    safe_error = sanitize_tool_result_for_log(repr(error))
    classification = classify_llm_exception(error, safe_error)
    provider_message = _exception_provider_message(error, safe_error)
    custody_fields = attempt_custody_event_fields(error)
    will_retry = classification.retry_same_request
    repeats = _transport_death_repeats(ctx.accumulated_usage, ctx.round_id)
    backoff = None
    if classification.kind == "provider_outcome_unknown":
        # Decided BEFORE the durable rows are written so `retry_same_request`
        # is the truth of what happens next: a bounded paid repeat (a NEW
        # attempt with its own ledger row) of a typed transport death while
        # this round's budget, the attempt loop AND the owner deadline (the
        # backoff plus the admission reserve the next iteration re-checks) all
        # have room; otherwise the no-resend terminal. Counted here, at the
        # grant, so the counter never names a repeat that was not sent.
        if (
            is_retryable_transport_death(error)
            and repeats < ctx.transport_death_retries and ctx.attempt < ctx.transient_budget - 1
        ):
            backoff = _TRANSPORT_DEATH_BACKOFF_SEC[min(repeats + 1, len(_TRANSPORT_DEATH_BACKOFF_SEC)) - 1]
            will_retry = not owner_deadline_exhausted(
                deadline_ts=ctx.deadline_ts,
                reserve_sec=backoff + max(ctx.transport_reserve_sec or 0.0, _DEADLINE_RETRY_FLOOR_SEC),
            )
    elif repeats and classification.kind != "transport_unavailable":
        # The fence, as a class: a round still holding an unresolved attempt (a repeat
        # was granted, no usable response since) sends nothing further whatever THIS
        # failure's class; a released ($0) repeat is the free wait episode's to redial.
        will_retry = False
    if will_retry and backoff is not None:
        ctx.accumulated_usage[TRANSPORT_DEATHS_KEY] = {
            "round_id": ctx.round_id, "count": repeats + 1, "backoff_sec": backoff,
        }
    elif repeats:
        # The granted repeat's own failure class (a grant writes no class; a later free redial finds it filled).
        ctx.accumulated_usage[TRANSPORT_DEATHS_KEY].setdefault("error_kind", classification.kind)
    identity = {
        "task_id": ctx.task_id, "execution_id": ctx.execution_id, "round_id": ctx.round_id,
        "llm_call_id": ctx.llm_call_id, "round": ctx.round_idx, "attempt": ctx.attempt + 1,
        "model": ctx.model,
    }
    _emit_live_log(ctx.event_queue, {
        "type": "llm_round_error", "task_type": ctx.task_type, **identity,
        "task_attempt": ctx.task_attempt, "error": safe_error,
        "error_kind": classification.kind, "retry_same_request": will_retry,
    })
    append_jsonl(ctx.drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "llm_api_error", **identity, "error": safe_error,
        "error_kind": classification.kind, "retry_same_request": will_retry,
        "status_code": classification.status_code, "provider_code": classification.provider_code,
        "provider_message": provider_message,
        **custody_fields,
        **(ctx.context_fit_event_fields or {}),
        "request_ref": ctx.request_ref.get("manifest_ref") if ctx.request_ref else None,
    })
    ctx.accumulated_usage.update(_last_llm_error=_short_error_text(safe_error),
                                 _last_llm_error_kind=classification.kind, _last_llm_retry_same_request=will_retry)
    if classification.retry_after_sec is not None:
        ctx.accumulated_usage["_last_llm_retry_after_sec"] = classification.retry_after_sec
        ctx.accumulated_usage["_last_llm_reset_at"] = classification.reset_at
    else:
        ctx.accumulated_usage.pop("_last_llm_retry_after_sec", None)
        ctx.accumulated_usage.pop("_last_llm_reset_at", None)
    for key, value in (("_last_llm_provider_message", provider_message), ("_last_llm_status_code", classification.status_code),
                       ("_last_llm_provider_code", classification.provider_code)):
        if value:
            ctx.accumulated_usage[key] = value
    ctx.accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
    if classification.kind == "context_overflow":
        overflow_event_type = "local_context_overflow" if isinstance(error, LocalContextTooLargeError) else "remote_context_overflow"
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": overflow_event_type, **identity, "error": safe_error,
            **(ctx.context_fit_event_fields or {}),
        })
        return True
    if not will_retry:
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "llm_non_retryable_same_request", **identity,
            "error_kind": classification.kind, "status_code": classification.status_code,
            "provider_code": classification.provider_code, "provider_message": provider_message,
        })
        return True
    return False


def _stop_after_llm_error(ctx: _LlmErrorContext) -> bool:
    """Retry decision for the exception path: ``True`` stops the attempt loop, and
    every stop except the two transport shapes stamps the OB-01 marker — permanent
    classes stopped earlier, so a stop here means "the same-model wall is spent"."""
    accumulated_usage = ctx.accumulated_usage
    error_kind = str(accumulated_usage.get("_last_llm_error_kind") or "")
    if error_kind == "transport_unavailable":
        # One free ($0) pre-dispatch attempt; the round-level wait episode owns
        # redial pacing. NOT a spent wall — the transport-wait terminal owns this.
        return True
    if error_kind == "provider_outcome_unknown":
        # Granted, counted and deadline-checked in _record_llm_call_error; only
        # the recorded backoff (by death ordinal) is left before the loop sends
        # a NEW physical attempt. Not a spent wall either — the unknown
        # no-resend terminal outranks the wall.
        backoff = (accumulated_usage.get(TRANSPORT_DEATHS_KEY) or {}).get("backoff_sec")
    else:
        is_transient = error_kind in _TRANSIENT_RETRY_KINDS
        # Non-transient retryables: max_retries capped by the loop ceiling (primary: no-op).
        attempt_budget = ctx.transient_budget if is_transient else min(ctx.max_retries, ctx.transient_budget)
        backoff = (
            _retry_backoff_sec(accumulated_usage, error_kind, ctx.attempt, is_transient)
            if ctx.attempt < attempt_budget - 1 else None
        )
    if backoff is not None:
        if _sleep_within_deadline(backoff, ctx.deadline_ts):
            return False
        if error_kind == "provider_outcome_unknown":
            _uncount_transport_death(accumulated_usage)  # the granted repeat never left the host
        _emit_retry_deadline_exhausted(
            ctx.drive_logs, task_id=ctx.task_id, execution_id=ctx.execution_id,
            round_id=ctx.round_id, round_idx=ctx.round_idx, attempt=ctx.attempt,
            model=ctx.model, error_kind=error_kind,
        )
    if error_kind != "provider_outcome_unknown":
        accumulated_usage[RETRY_WALL_EXHAUSTED_KEY] = True
    return True


def _empty_response_wall_spent(is_provider_glitch: bool, permanent_body_error: bool, usage: Dict[str, Any]) -> bool:
    """Empty-response exit: spent only while the PROVIDER is failing (finish=None
    glitch or transient body error); a permanent body error and a live provider
    (finish="stop"/"length") both keep their one forced call."""
    return not permanent_body_error and (is_provider_glitch or bool(usage.get("provider_error")))


def provider_no_call_source(accumulated_usage: Dict[str, Any], deadline_exhausted: bool) -> Tuple[str, bool]:
    """The provider-unavailable rail's no-call decision → (typed source, wall_spent).
    Unknown in-flight outcome forbids a RESEND (outranks the wall); a SPENT same-model
    wall makes one more forced call a second full retry window; the deadline_local
    rail keeps its grace call, so ``deadline_exhausted`` suppresses the wall. A round
    still holding an unresolved attempt (its transport-death record: written at a grant,
    cleared only by a usable response) forbids the resend whatever the sticky kind."""
    if str(accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown" or isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
        return "provider_outcome_unknown_no_resend", False
    if not deadline_exhausted and bool(accumulated_usage.get(RETRY_WALL_EXHAUSTED_KEY)):
        return "retry_wall_exhausted_no_repay", True
    return "", False


def _emit_empty_response_events(
    event_type: str,
    *,
    event_queue: Optional[queue.Queue],
    drive_logs: pathlib.Path,
    base: Dict[str, Any],
    task_type: str,
    details: Dict[str, Any],
) -> None:
    """Emit the live log + durable event for an empty/incomplete LLM response.

    ``details`` carries the durable-event-only payload: content, tool_calls,
    request_ref, response_ref.
    """
    content = details.get("content")
    tool_calls = details.get("tool_calls")
    _emit_live_log(event_queue, {"type": event_type, "task_type": task_type, **base})
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": event_type,
        **base,
        "raw_content": repr(content)[:500] if content else None,
        "raw_tool_calls": repr(tool_calls)[:500] if tool_calls else None,
        "request_ref": (details.get("request_ref") or {}).get("manifest_ref"),
        "response_ref": (details.get("response_ref") or {}).get("manifest_ref"),
    })


def _context_fit_event_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "context_route_fp": str(usage.get("_context_route_fp") or ""),
        "estimated_prompt_tokens": int(usage.get("_context_prompt_estimate") or 0),
        "context_fit_mode": str(usage.get("_context_fit_mode") or ""),
        "context_profile": str(usage.get("_context_profile") or ""),
        "context_measurement_basis": str(usage.get("_context_measurement_basis") or ""),
        "context_measurement_density": float(usage.get("_context_measurement_density") or 0.0),
        "context_target_total_tokens": usage.get("_context_target_total_tokens"),
        "context_capacity_total_tokens": usage.get("_context_capacity_total_tokens"),
        "context_target_deficit_tokens": usage.get("_context_target_deficit_tokens"),
        "context_capacity_deficit_tokens": usage.get("_context_capacity_deficit_tokens"),
        "context_reclaim_goal_tokens": int(usage.get("_context_reclaim_goal_tokens") or 0),
        "context_target_miss": bool(usage.get("_context_target_miss")),
        "context_automatic_pass_used": bool(usage.get("_context_automatic_pass_used")),
        "context_predicted_capacity_miss": bool(
            usage.get("_context_predicted_capacity_miss")
        ),
    }


def _record_round_cache_facts(
    accumulated_usage: Dict[str, Any],
    usage: Dict[str, Any],
    *,
    round_idx: int,
) -> tuple:
    """Cold-restart telemetry — FACTS of this round only (no dollar counterfactuals).

    Returns ``(prompt_cache_ttl, cache_hit_rate, cache_cold_restart,
    gap_since_prev_round_sec)``: a later round whose prompt was almost entirely
    re-written is a provider-cache expiry/invalidation made structurally visible,
    and the gap since the previous successful round is the datum that separates a
    TTL expiry (long wait) from a prefix rewrite. Also records the APPLIED TTL of
    this task's latest send — the recorded fact the wait-tool cache-horizon
    disclosure reads (one place, never re-derived).
    """
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    cache_write_tokens = int(usage.get("cache_write_tokens") or 0)
    prompt_cache_ttl = str(usage.get("prompt_cache_ttl") or "")
    cache_hit_rate = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
    prev_finished = accumulated_usage.get("_last_llm_round_finished_monotonic")
    gap_since_prev_round_sec = (
        round(max(0.0, time.monotonic() - float(prev_finished)), 1)
        if isinstance(prev_finished, (int, float)) else None
    )
    cache_cold_restart = bool(
        round_idx > 1
        and prompt_tokens > 0
        and cached_tokens < 0.2 * prompt_tokens
        and cache_write_tokens > 0.5 * prompt_tokens
    )
    accumulated_usage["_last_llm_round_finished_monotonic"] = time.monotonic()
    accumulated_usage["_last_prompt_cache_ttl"] = prompt_cache_ttl
    return prompt_cache_ttl, cache_hit_rate, cache_cold_restart, gap_since_prev_round_sec


def _prepare_main_messages(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    llm: LLMClient,
    accumulated_usage: Dict[str, Any],
    drive_root: pathlib.Path,
    task_id: str,
    event_queue: Optional[queue.Queue],
    use_local: bool,
    task_attempt: Any = None,
    deadline_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    try:
        from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

        return prepare_messages_for_send(
            messages,
            routing=VisionRoutingContext(
                model=model, llm=llm, accumulated_usage=accumulated_usage,
                drive_root=drive_root, task_id=task_id, event_queue=event_queue,
                use_local=use_local, task_attempt=task_attempt, deadline_ts=deadline_ts,
            ),
        )
    except Exception:
        log.debug("vision routing preparation failed; falling back to canonical messages", exc_info=True)
        return messages


def _send_main_candidate(
    llm: LLMClient,
    kwargs: Dict[str, Any],
    *,
    model: str,
    use_local: bool,
    deadline_ts: Optional[float],
    physical_context: Optional[PhysicalAttemptContext],
    candidate_predicate: Optional[Callable[[Any], Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    binding = (
        contextlib.nullcontext() if physical_context is None and candidate_predicate is None
        else bind_physical_attempt_context(physical_context, candidate_predicate=candidate_predicate)
    )
    with model_concurrency.model_call_slot(model, use_local, deadline_ts), binding:
        return llm.chat(**kwargs)


def _take_custom_receipts(
    usage: Dict[str, Any],
    msg: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> None:
    from ouroboros.openai_chat_dispatch import (
        CUSTOM_RECEIPTS_USAGE_KEY,
        pop_custom_validation_receipts,
    )
    receipts = pop_custom_validation_receipts(usage, msg.get("tool_calls") or [])
    accumulated_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
    if receipts:
        accumulated_usage[CUSTOM_RECEIPTS_USAGE_KEY] = receipts


def _emit_main_llm_call_state(
    event_queue: Optional[queue.Queue],
    identity: Tuple[Any, ...],
    phase: str,
) -> None:
    task_id, task_attempt, llm_call_id, execution_id, round_id, call_attempt = identity
    emit_main_llm_call_state_event(
        event_queue,
        task_id=task_id,
        task_attempt=task_attempt,
        llm_call_id=llm_call_id,
        execution_id=execution_id,
        round_id=round_id,
        call_attempt=call_attempt,
        phase=phase,
    )


def _handle_main_llm_call_exception(
    error: Exception, ctx: _LlmErrorContext, call_identity: Tuple[Any, ...],
) -> bool:
    """Close the exact call and decide whether the attempt loop must stop."""
    from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY
    _emit_main_llm_call_state(ctx.event_queue, call_identity, "failed")
    _emit_llm_operation(
        ctx.event_queue, ctx.task_id, ctx.llm_call_id, "failed", ctx.task_attempt,
        ctx.execution_id, ctx.round_id,
    )
    ctx.accumulated_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
    return _record_llm_call_error(error, ctx) or _stop_after_llm_error(ctx)


def _replace_response_meta(
    target: Optional[Dict[str, Any]],
    usage: Optional[Dict[str, Any]] = None,
    msg: Optional[Dict[str, Any]] = None,
) -> None:
    """Replace one caller-owned, attempt-local response fact projection."""
    if target is None:
        return
    target.clear()
    if usage is None or msg is None:
        return
    for source, key in ((usage, "response_finish_reason"), (msg, "finish_reason"), (msg, "stop_reason")):
        if key in source:
            finish_present, finish_reason = True, source.get(key)
            break
    else:
        finish_present, finish_reason = False, None
    target.update({
        "finish_reason_present": finish_present,
        "finish_reason": finish_reason,
        "tool_call_count": len(msg.get("tool_calls") or []),
    })


def forced_response_parts(
    result: Any, accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Normalize the forced-call seam while preserving legacy tuple patches."""
    if isinstance(result, tuple):
        return (
            str(result[0] or "").strip(),
            dict(result[1]) if len(result) > 1 and isinstance(result[1], dict) else {},
        )
    meta = accumulated_usage.pop("_forced_response_meta", {})
    return str(result or "").strip(), dict(meta) if isinstance(meta, dict) else {}


def _emit_llm_operation(
    event_queue: Optional[queue.Queue], task_id: str, operation_id: str,
    phase: str, task_attempt: Any, execution_id: str, round_id: str,
) -> None:
    emit_cognitive_operation_event(
        event_queue, task_id=task_id, operation_id=operation_id, phase=phase,
        kind="llm", task_attempt=task_attempt, execution_id=execution_id,
        round_id=round_id,
    )


def call_llm_with_retry(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    effort: str,
    max_retries: int,
    drive_logs: pathlib.Path,
    task_id: str,
    round_idx: int,
    event_queue: Optional[queue.Queue],
    accumulated_usage: Dict[str, Any],
    task_type: str = "",
    use_local: bool = False,
    deadline_ts: Optional[float] = None,
    attempt_cap: Optional[int] = None,
    allow_server_web_search: bool = False,
    physical_context: Optional[PhysicalAttemptContext] = None,
    candidate_predicate: Optional[Callable[[Any], Any]] = None,
    task_attempt: Any = None,
    response_meta_out: Optional[Dict[str, Any]] = None,
    transport_reserve_sec: Optional[float] = None, transport_death_retries: int = 0,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Call one model with bounded retries and deadline-aware transport."""
    msg = None
    _replace_response_meta(response_meta_out)
    drive_root = pathlib.Path(drive_logs).parent
    accumulated_usage.pop(RETRY_WALL_EXHAUSTED_KEY, None)  # last-invocation marker (see key)
    execution_id = str(accumulated_usage.setdefault("execution_id", new_execution_id()))
    round_id = f"{execution_id}:round:{round_idx}"
    _transport_death_repeats(accumulated_usage, round_id)  # drops another round's record before anything reads it
    context_fit_event_fields = _context_fit_event_fields(accumulated_usage) if physical_context is not None else {}
    transient_budget = _attempt_loop_budget(max_retries, attempt_cap)
    if task_attempt is None:
        task_attempt = accumulated_usage.get("_task_attempt")
    else:
        accumulated_usage["_task_attempt"] = task_attempt
    response_cache_bypass_requested = False
    for attempt in range(transient_budget):
        if _deadline_not_dispatched(
            deadline_ts, accumulated_usage, drive_logs,
            task_id=task_id, model=model, round_idx=round_idx, round_id=round_id, reserve_sec=transport_reserve_sec,
        ):
            return None, None
        llm_call_id = new_call_id("llm")
        call_identity = (task_id, task_attempt, llm_call_id, execution_id, round_id, attempt + 1)
        request_ref: Dict[str, Any] = {}
        try:
            _emit_llm_operation(event_queue, task_id, llm_call_id, "started", task_attempt, execution_id, round_id)
            send_messages = initial_messages if attempt == 0 and initial_messages is not None else _prepare_main_messages(
                messages, model=model, llm=llm, accumulated_usage=accumulated_usage,
                drive_root=drive_root, task_id=task_id, event_queue=event_queue,
                use_local=use_local, task_attempt=task_attempt, deadline_ts=deadline_ts,
            )
            _emit_live_log(event_queue, {
                "type": "llm_round_started",
                "task_id": task_id,
                "task_type": task_type,
                "execution_id": execution_id,
                "round_id": round_id,
                "llm_call_id": llm_call_id,
                "round": round_idx,
                "task_attempt": task_attempt,
                "attempt": attempt + 1,
                "model": model,
                "reasoning_effort": effort,
                "use_local": bool(use_local),
            })
            kwargs = {
                "messages": send_messages,
                "model": model,
                "reasoning_effort": effort,
                "max_tokens": MAIN_LOOP_MAX_TOKENS,
                "use_local": use_local,
                "allow_server_web_search": bool(allow_server_web_search),
                "bypass_response_cache": response_cache_bypass_requested,
                "timeout": _main_transport_timeout(
                    model, deadline_ts, reserve_sec=transport_reserve_sec,
                ),
            }
            if tools:
                kwargs["tools"] = tools
            try:
                request_ref = persist_call(
                    drive_root,
                    task_id=task_id,
                    call_id=f"{llm_call_id}_request",
                    call_type="llm_request",
                    payload=public_custody_projection({
                        "messages": messages,
                        "send_messages": send_messages,
                        "tools": tools or [],
                        "model": model,
                        "reasoning_effort": effort,
                        "max_tokens": MAIN_LOOP_MAX_TOKENS,
                        "use_local": bool(use_local),
                        "allow_server_web_search": bool(allow_server_web_search),
                        "response_cache_bypass_requested": response_cache_bypass_requested,
                    }),
                    manifest={
                        "execution_id": execution_id,
                        "round_id": round_id,
                        "llm_call_id": llm_call_id,
                        "round": round_idx,
                        "attempt": attempt + 1,
                        "model": model,
                        "reasoning_effort": effort,
                        "response_cache_bypass_requested": response_cache_bypass_requested,
                        **context_fit_event_fields,
                    },
                )
            except Exception:
                log.debug("Failed to persist LLM request observability payload", exc_info=True)
            if _deadline_not_dispatched(
                deadline_ts, accumulated_usage, drive_logs,
                task_id=task_id, model=model, round_idx=round_idx,
                event_queue=event_queue, llm_call_id=llm_call_id,
                task_attempt=task_attempt, execution_id=execution_id,
                round_id=round_id, reserve_sec=transport_reserve_sec,
            ):
                return None, None
            _emit_main_llm_call_state(event_queue, call_identity, "started")
            resp_msg, usage = _send_main_candidate(
                llm, kwargs, model=model, use_local=use_local, deadline_ts=deadline_ts,
                physical_context=physical_context, candidate_predicate=candidate_predicate if attempt == 0 else None,
            )
            msg = resp_msg
            _take_custom_receipts(usage, msg, accumulated_usage)
            for stale in ("_last_llm_error", "_last_llm_error_kind", "_last_llm_retry_same_request",
                          "_last_llm_status_code", "_last_llm_provider_code"):
                accumulated_usage.pop(stale, None)
            cost, display_model, provider, cost_estimated = _normalize_usage_cost(
                usage,
                model=model,
                use_local=use_local,
            )
            add_usage(accumulated_usage, usage)
            fold_retrieval_usage(accumulated_usage, usage)
            response_ref: Dict[str, Any] = {}
            try:
                response_ref = persist_call(
                    drive_root,
                    task_id=task_id,
                    call_id=f"{llm_call_id}_response",
                    call_type="llm_response",
                    payload=public_custody_projection({
                        "message": msg,
                        "usage": usage,
                    }),
                    manifest={
                        "execution_id": execution_id,
                        "round_id": round_id,
                        "llm_call_id": llm_call_id,
                        "round": round_idx,
                        "attempt": attempt + 1,
                        "model": model,
                        "resolved_model": display_model,
                        "provider": provider,
                    },
                )
            except Exception:
                log.debug("Failed to persist LLM response observability payload", exc_info=True)
            _remember_llm_call(
                accumulated_usage,
                llm_call_id=llm_call_id,
                execution_id=execution_id,
                round_id=round_id,
                round_idx=round_idx,
                attempt=attempt + 1,
                model=model,
                display_model=display_model,
                provider=provider,
                request_ref=request_ref,
                response_ref=response_ref,
            )
            category = task_type if task_type in ("evolution", "consciousness", "review", "summarize") else "task"
            emit_llm_usage_event(
                event_queue,
                task_id,
                display_model,
                {**usage, "llm_call_id": llm_call_id, "execution_id": execution_id,
                 "round_id": round_id, "round": round_idx},
                cost,
                category,
                provider=provider,
                source="loop",
                cost_estimated=cost_estimated,
            )
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            _replace_response_meta(response_meta_out, usage, msg)
            if not tool_calls and (not content or not content.strip()):
                event_type, is_provider_glitch, permanent_body_error = _record_and_emit_empty_response(
                    usage=usage, msg=msg, accumulated_usage=accumulated_usage,
                    event_queue=event_queue, drive_logs=drive_logs, task_id=task_id,
                    execution_id=execution_id, round_id=round_id, llm_call_id=llm_call_id,
                    round_idx=round_idx, attempt=attempt, model=model, task_type=task_type,
                    content=content, tool_calls=tool_calls, request_ref=request_ref,
                    response_ref=response_ref, transient_budget=transient_budget,
                    context_fit_event_fields=context_fit_event_fields, task_attempt=task_attempt)
                _emit_llm_operation(event_queue, task_id, llm_call_id, "failed", task_attempt, execution_id, round_id)
                _emit_main_llm_call_state(event_queue, call_identity, "failed")
                if event_type == "provider_incomplete_response" and not usage.get("provider_error"):
                    response_cache_bypass_requested = True
                # fenced like any other class while the round holds an unresolved attempt
                if not permanent_body_error and attempt < transient_budget - 1 and TRANSPORT_DEATHS_KEY not in accumulated_usage:
                    if _sleep_within_deadline(
                        min(2.0 ** attempt, _TRANSIENT_BACKOFF_CAP_SEC), deadline_ts
                    ):
                        continue
                    _emit_retry_deadline_exhausted(
                        drive_logs, task_id=task_id, execution_id=execution_id,
                        round_id=round_id, round_idx=round_idx, attempt=attempt,
                        model=model, error_kind=event_type,
                    )
                if _empty_response_wall_spent(is_provider_glitch, permanent_body_error, usage):
                    accumulated_usage[RETRY_WALL_EXHAUSTED_KEY] = True
                return None, cost
            for stale in ("execution_status", "result_status", "reason_code", RETRY_WALL_EXHAUSTED_KEY, TRANSPORT_DEATHS_KEY):
                accumulated_usage.pop(stale, None)  # a USABLE response closes the round's repeat record
            accumulated_usage["rounds"] = accumulated_usage.get("rounds", 0) + 1
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            cached_tokens = int(usage.get("cached_tokens") or 0)
            cache_write_tokens = int(usage.get("cache_write_tokens") or 0)
            prompt_cache_ttl, cache_hit_rate, cache_cold_restart, gap_since_prev_round_sec = (
                _record_round_cache_facts(accumulated_usage, usage, round_idx=round_idx))
            _round_event = {
                "ts": utc_now_iso(), "type": "llm_round",
                "task_id": task_id,
                "execution_id": execution_id,
                "round_id": round_id,
                "llm_call_id": llm_call_id,
                "round": round_idx, "model": display_model,
                "reasoning_effort": effort,
                "provider": provider,
                "source": "loop",
                "model_category": infer_model_category(display_model),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
                "prompt_cache_ttl": prompt_cache_ttl,
                "cache_hit_rate": cache_hit_rate,
                "cache_cold_restart": cache_cold_restart,
                "gap_since_prev_round_sec": gap_since_prev_round_sec,
                "cost_usd": cost,
                **context_fit_event_fields,
                "request_ref": request_ref.get("manifest_ref") if request_ref else None,
                "response_ref": response_ref.get("manifest_ref") if response_ref else None,
            }
            _emit_live_log(event_queue, {
                "type": "llm_round_finished",
                "task_id": task_id,
                "task_type": task_type,
                "execution_id": execution_id,
                "round_id": round_id,
                "llm_call_id": llm_call_id,
                "round": round_idx,
                "task_attempt": task_attempt,
                "attempt": attempt + 1,
                "model": display_model,
                "reasoning_effort": effort,
                **{key: _round_event[key] for key in (
                    "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens", "prompt_cache_ttl")},
                "cost_usd": cost,
                "response_kind": "tool_calls" if tool_calls else "message",
                "tool_call_count": len(tool_calls),
                "has_text": bool(content and str(content).strip()),
            })
            append_jsonl(drive_logs / "events.jsonl", _round_event)
            _emit_llm_operation(event_queue, task_id, llm_call_id, "finished", task_attempt, execution_id, round_id)
            _emit_main_llm_call_state(event_queue, call_identity, "finished")
            return msg, cost
        except UsageAccountingError:
            _emit_llm_operation(event_queue, task_id, llm_call_id, "failed", task_attempt, execution_id, round_id)
            _emit_main_llm_call_state(event_queue, call_identity, "failed")
            raise  # Monetary/ledger rails are not provider failures.
        except Exception as e:
            if _handle_main_llm_call_exception(
                e,
                _LlmErrorContext(
                    task_id=task_id, task_type=task_type, execution_id=execution_id,
                    round_id=round_id, llm_call_id=llm_call_id, round_idx=round_idx,
                    attempt=attempt, model=model, request_ref=request_ref,
                    drive_logs=drive_logs, event_queue=event_queue,
                    accumulated_usage=accumulated_usage,
                    context_fit_event_fields=context_fit_event_fields,
                    task_attempt=task_attempt, deadline_ts=deadline_ts,
                    max_retries=max_retries, transient_budget=transient_budget,
                    transport_death_retries=transport_death_retries, transport_reserve_sec=transport_reserve_sec,
                ),
                call_identity,
            ):
                break
    return None, 0.0
