"""
LLM call, retry, pricing, and usage-event logic for the main loop.

Handles model pricing estimation, cost tracking, per-call retry with backoff,
and real-time usage event emission.
Extracted from loop.py to keep the main loop orchestrator focused.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros import model_concurrency
from ouroboros.deadline_utils import seconds_until
from ouroboros.llm import LLMClient, LocalContextTooLargeError, add_usage
from ouroboros.llm_attempt import (  # the typed-refusal contract's owner, not a copy of it
    PROVIDER_POLICY_REFUSAL,
    _is_provider_policy_refusal,
)
from ouroboros.observability import new_call_id, new_execution_id, persist_call
from ouroboros.pricing import emit_llm_usage_event, estimate_cost_optional, infer_model_category
from ouroboros.usage_accounting import (
    PhysicalAttemptContext,
    UsageAccountingError,
    bind_physical_attempt_context,
)
from ouroboros.utils import append_jsonl, emit_log_event, sanitize_tool_result_for_log, truncate_review_artifact, utc_now_iso

log = logging.getLogger(__name__)

MAIN_LOOP_MAX_TOKENS = 65_536

# Retrieval transparency (v6.78.0, owner Q20/Q22): native provider web search happens
# INSIDE the solve model's own request, so `usage["web_search_sources"]` /
# `usage["server_tool_use"]` are the only host-attested evidence that the answer was
# grounded in fetched pages. `add_usage` accumulates numeric token keys only, so
# without this fold the fact dies at the per-call boundary and the acceptance reviewer
# never learns whether an answer came from retrieval or from the model's own knowledge.
# Counts plus capped URLs only — no titles, no snippets (leak-safe, bounded).
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
    # Dedup keys over the FULL RAW url, one per RETAINED entry (so this stays O(cap)
    # memory like the rest of the accumulator). Round 3: deduping on the RENDERED value
    # silently lost evidence — two DISTINCT long URLs sharing the retained prefix and
    # length render byte-identically, so the second was skipped while `urls_omitted`
    # still reported 0, i.e. a fetched URL vanished from evidence that promises an exact
    # omission count (BIBLE P1). Raw keys make a repeat a true repeat and every distinct
    # URL either carried or counted.
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


def _empty_response_log_msg(usage: Dict[str, Any], is_provider_glitch: bool, accumulated_usage: Dict[str, Any]) -> str:
    """Honest message for an empty/incomplete LLM response. A transient provider
    body-error (OpenRouter 429/5xx inside an HTTP 200, surfaced as usage
    ``provider_error``) that a same-model reroute could not escape is classified
    as the real rate_limit/provider_transient kind instead of a blank
    finish_reason=null glitch."""
    provider_error = usage.get("provider_error") if isinstance(usage, dict) else None
    if isinstance(provider_error, dict):
        accumulated_usage["_last_llm_error_kind"] = str(provider_error.get("kind") or "provider_transient")
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
) -> tuple:
    """Classify an empty / no-tool-call response, log + emit its events, and stamp
    accumulated_usage (last error / execution_status / reason_code / F1 cooldown kind).
    Returns ``(event_type, is_provider_glitch, permanent_body_error)`` for the caller's
    retry decision. Extracted from call_llm_with_retry to keep that loop readable."""
    finish_reason = msg.get("finish_reason") or msg.get("stop_reason")
    event_type, is_provider_glitch, permanent_body_error = _classify_empty_response(usage, msg)
    log_msg = _empty_response_log_msg(usage, is_provider_glitch, accumulated_usage)
    log.warning("%s, attempt %d/%d", log_msg, attempt + 1, transient_budget)
    _emit_empty_response_events(
        event_type, event_queue=event_queue, drive_logs=drive_logs,
        base={"task_id": task_id, "execution_id": execution_id, "round_id": round_id,
              "llm_call_id": llm_call_id, "round": round_idx, "attempt": attempt + 1,
              "model": model, "finish_reason": finish_reason,
              **context_fit_event_fields},
        task_type=task_type,
        details={"content": content, "tool_calls": tool_calls,
                 "request_ref": request_ref, "response_ref": response_ref},
    )
    accumulated_usage["_last_llm_error"] = _short_error_text(log_msg)
    if event_type == "remote_context_overflow":
        accumulated_usage["execution_status"] = "infra_failed"
        accumulated_usage["reason_code"] = "llm_api_error"
        accumulated_usage["_last_llm_error_kind"] = "context_overflow"
    else:
        accumulated_usage["execution_status"] = (
            "infra_failed" if (is_provider_glitch and not permanent_body_error) else "failed"
        )
        accumulated_usage["reason_code"] = event_type
        # Cooldown signal for the F1 fallback gate (see helper; not a retry change).
        accumulated_usage["_last_llm_error_kind"] = _cooldown_kind_for_empty_response(
            usage, event_type,
        )
    return event_type, is_provider_glitch, permanent_body_error


def _cooldown_kind_for_empty_response(usage: Dict[str, Any], event_type: str) -> str:
    """Pick the kind exposed as ``_last_llm_error_kind`` for the F1 fallback-chain cooldown
    gate on an empty/body-error response. PREFER the provider body-error kind (a 429
    surfaces as ``rate_limit``) so a rate-limited model cools down regardless of
    finish_reason; otherwise fall back to ``event_type`` (``provider_incomplete_response``
    cools; ``provider_body_error`` / ``llm_empty_response`` are not in the cooldown set, so
    they correctly do not). This is purely the cooldown SIGNAL — it does not change the
    same-model transient-retry layering (the primary keeps its full plan-preserved budget;
    cooldown is the second layer once that budget is exhausted)."""
    body_err = usage.get("provider_error") if isinstance(usage, dict) else None
    body_kind = str((body_err or {}).get("kind") or "") if isinstance(body_err, dict) else ""
    if isinstance(body_err, dict):
        body_code = str(body_err.get("code") or "").strip()
        body_message = str(body_err.get("message") or "")
        if body_code == "413" or _is_output_or_body_size_error(body_message):
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
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


from ouroboros.context_budget import (  # one overflow vocabulary for every seam
    CONTEXT_OVERFLOW_CODES as _STRUCTURED_CONTEXT_OVERFLOW_CODES,
    context_overflow_message as _context_overflow_message,
    output_or_body_size_message as _output_or_body_size_message,
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


def _is_rate_limit_text(text: str) -> bool:
    low = str(text or "").lower()
    return any(marker in low for marker in _RATE_LIMIT_TEXT_MARKERS)


def _is_context_overflow_error(exc: Exception, safe_error: str) -> bool:
    """Classify untyped local/remote context-window overflow.

    The output-size precedence lives inside the SHARED helper, not here, so
    Main, the local transport, and the summarizer classify identically."""
    if isinstance(exc, LocalContextTooLargeError):
        return True
    if _is_rate_limit_text(safe_error):
        return False
    return _context_overflow_message(safe_error)


def _is_output_or_body_size_error(safe_error: str) -> bool:
    return _output_or_body_size_message(safe_error)


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
    return value if isinstance(value, int) else None


def _exception_provider_values(exc: Exception) -> List[str]:
    values: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)

    for attr in ("code", "type"):
        add(getattr(exc, attr, None))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        for key in ("code", "type"):
            add(body.get(key))
        nested = body.get("error")
        if isinstance(nested, dict):
            for key in ("code", "type"):
                add(nested.get(key))
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
    return values[0] if values else ""


def _exception_provider_message(exc: Exception, safe_error: str = "") -> str:
    """Best-effort human-readable provider error BODY text.

    Strict OpenAI-compatible providers (cloud.ru Foundation Models, vLLM/SGLang)
    return a 400 whose BODY distinguishes otherwise-identical status codes — e.g. a
    cloud.ru content-filter ("guardrails") block vs an ``Extra inputs are not
    permitted`` reasoning_content echo. ``provider_code`` alone cannot tell them
    apart, so surface the body message (sanitized + truncated) into the durable
    event for the owner. Pure read of ``exc.body``/repr; never changes routing."""
    body = getattr(exc, "body", None)
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
        if code == kind or any(code == str(marker).lower() or str(marker).lower() in code for marker in markers):
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
            SUBSCRIPTION_WINDOW_EXHAUSTED,
            True,
            _exception_status_code(exc),
            "",
            seconds_until(reset_at),
            reset_at,
        )
    status_code = _exception_status_code(exc)
    provider_code = _exception_provider_code(exc, safe)
    # Same structured contract, a different fact: a policy layer would not let this
    # call reach a provider at all. Nothing upstream answered, so retrying the
    # UNCHANGED request only re-runs the refusal — the class is permanent by type,
    # exactly as the recovery ladder already treats it (llm_attempt.
    # ProviderPolicyRefusal). Read through the owner's predicate so class and
    # declared-code shapes stay one contract, and so it outranks every heuristic
    # below: a refusal carries no provider prose worth guessing from.
    if _is_provider_policy_refusal(exc):
        return LlmErrorClassification(
            PROVIDER_POLICY_REFUSAL,
            False,
            status_code,
            provider_code or PROVIDER_POLICY_REFUSAL,
        )
    low = str(safe or "").lower()
    if provider_code.lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
        return LlmErrorClassification("context_overflow", False, status_code, provider_code)
    provider_kind = _provider_code_kind(provider_code)
    if provider_kind:
        return LlmErrorClassification(provider_kind, False, status_code, provider_code)
    if status_code == 429:
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if _is_rate_limit_text(low):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if _is_output_or_body_size_error(low):
        return LlmErrorClassification("request_too_large", False, status_code, provider_code)
    if _is_context_overflow_error(exc, safe):
        return LlmErrorClassification("context_overflow", False, status_code, provider_code)
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
    if status_code in {408, 500, 502, 503, 504}:
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
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
    provider_reported_cost = usage.get("cost") is not None
    cost = float(usage["cost"]) if provider_reported_cost else None
    display_model = str(usage.get("resolved_model") or model)
    provider = "local" if use_local else str(usage.get("provider") or "openrouter")
    if use_local:
        cost = 0.0
        display_model = f"{model} (local)"
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
    _emit_live_log(ctx.event_queue, {
        "type": "llm_round_error",
        "task_id": ctx.task_id,
        "task_type": ctx.task_type,
        "execution_id": ctx.execution_id,
        "round_id": ctx.round_id,
        "llm_call_id": ctx.llm_call_id,
        "round": ctx.round_idx,
        "attempt": ctx.attempt + 1,
        "model": ctx.model,
        "error": safe_error,
        "error_kind": classification.kind,
        "retry_same_request": classification.retry_same_request,
    })
    append_jsonl(ctx.drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "llm_api_error",
        "task_id": ctx.task_id,
        "execution_id": ctx.execution_id,
        "round_id": ctx.round_id,
        "llm_call_id": ctx.llm_call_id,
        "round": ctx.round_idx, "attempt": ctx.attempt + 1,
        "model": ctx.model, "error": safe_error,
        "error_kind": classification.kind,
        "retry_same_request": classification.retry_same_request,
        "status_code": classification.status_code,
        "provider_code": classification.provider_code,
        "provider_message": provider_message,
        **(ctx.context_fit_event_fields or {}),
        "request_ref": ctx.request_ref.get("manifest_ref") if ctx.request_ref else None,
    })
    ctx.accumulated_usage["_last_llm_error"] = _short_error_text(safe_error)
    if provider_message:
        ctx.accumulated_usage["_last_llm_provider_message"] = provider_message
    ctx.accumulated_usage["_last_llm_error_kind"] = classification.kind
    ctx.accumulated_usage["_last_llm_retry_same_request"] = classification.retry_same_request
    if classification.retry_after_sec is not None:
        ctx.accumulated_usage["_last_llm_retry_after_sec"] = classification.retry_after_sec
        ctx.accumulated_usage["_last_llm_reset_at"] = classification.reset_at
    else:
        ctx.accumulated_usage.pop("_last_llm_retry_after_sec", None)
        ctx.accumulated_usage.pop("_last_llm_reset_at", None)
    if classification.status_code:
        ctx.accumulated_usage["_last_llm_status_code"] = classification.status_code
    if classification.provider_code:
        ctx.accumulated_usage["_last_llm_provider_code"] = classification.provider_code
    ctx.accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
    if classification.kind == "context_overflow":
        overflow_event_type = "local_context_overflow" if isinstance(error, LocalContextTooLargeError) else "remote_context_overflow"
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": overflow_event_type,
            "task_id": ctx.task_id,
            "execution_id": ctx.execution_id,
            "round_id": ctx.round_id,
            "llm_call_id": ctx.llm_call_id,
            "round": ctx.round_idx,
            "attempt": ctx.attempt + 1,
            "model": ctx.model,
            "error": safe_error,
            **(ctx.context_fit_event_fields or {}),
        })
        return True
    if not classification.retry_same_request:
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "llm_non_retryable_same_request",
            "task_id": ctx.task_id,
            "execution_id": ctx.execution_id,
            "round_id": ctx.round_id,
            "llm_call_id": ctx.llm_call_id,
            "round": ctx.round_idx,
            "attempt": ctx.attempt + 1,
            "model": ctx.model,
            "error_kind": classification.kind,
            "status_code": classification.status_code,
            "provider_code": classification.provider_code,
            "provider_message": provider_message,
        })
        return True
    return False


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
    request_ref = details.get("request_ref") or {}
    response_ref = details.get("response_ref") or {}
    _emit_live_log(event_queue, {"type": event_type, "task_type": task_type, **base})
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": event_type,
        **base,
        "raw_content": repr(content)[:500] if content else None,
        "raw_tool_calls": repr(tool_calls)[:500] if tool_calls else None,
        "request_ref": request_ref.get("manifest_ref") if request_ref else None,
        "response_ref": response_ref.get("manifest_ref") if response_ref else None,
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
) -> List[Dict[str, Any]]:
    try:
        from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

        return prepare_messages_for_send(
            messages,
            routing=VisionRoutingContext(
                model=model, llm=llm, accumulated_usage=accumulated_usage,
                drive_root=drive_root, task_id=task_id, event_queue=event_queue,
                use_local=use_local,
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
    with model_concurrency.model_call_slot(model, use_local, deadline_ts):
        if physical_context is None:
            return llm.chat(**kwargs)
        with bind_physical_attempt_context(
            physical_context, candidate_predicate=candidate_predicate,
        ):
            return llm.chat(**kwargs)


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
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Call one model with failure-class retry budgets and usage events.

    ``deadline_ts`` bounds backoff, ``attempt_cap`` caps fallback candidates,
    and cross-model fallback remains the caller's responsibility.
    """
    msg = None
    drive_root = pathlib.Path(drive_logs).parent
    execution_id = str(accumulated_usage.setdefault("execution_id", new_execution_id()))
    round_id = f"{execution_id}:round:{round_idx}"
    context_fit_event_fields = (
        _context_fit_event_fields(accumulated_usage)
        if physical_context is not None
        else {}
    )
    transient_budget = _attempt_loop_budget(max_retries, attempt_cap)
    response_cache_bypass_requested = False
    for attempt in range(transient_budget):
        accumulated_usage["_llm_attempts_used"] = attempt + 1
        llm_call_id = new_call_id("llm")
        request_ref: Dict[str, Any] = {}
        try:
            send_messages = _prepare_main_messages(
                messages, model=model, llm=llm, accumulated_usage=accumulated_usage,
                drive_root=drive_root, task_id=task_id, event_queue=event_queue,
                use_local=use_local,
            )
            _emit_live_log(event_queue, {
                "type": "llm_round_started",
                "task_id": task_id,
                "task_type": task_type,
                "execution_id": execution_id,
                "round_id": round_id,
                "llm_call_id": llm_call_id,
                "round": round_idx,
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
            }
            if tools:
                kwargs["tools"] = tools
            try:
                request_ref = persist_call(
                    drive_root,
                    task_id=task_id,
                    call_id=f"{llm_call_id}_request",
                    call_type="llm_request",
                    payload={
                        "messages": messages,
                        "send_messages": send_messages,
                        "tools": tools or [],
                        "model": model,
                        "reasoning_effort": effort,
                        "max_tokens": MAIN_LOOP_MAX_TOKENS,
                        "use_local": bool(use_local),
                        "allow_server_web_search": bool(allow_server_web_search),
                        "response_cache_bypass_requested": response_cache_bypass_requested,
                    },
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
            # Vision preparation is outside the Main-only physical binding.
            resp_msg, usage = _send_main_candidate(
                llm, kwargs, model=model, use_local=use_local, deadline_ts=deadline_ts,
                physical_context=physical_context, candidate_predicate=candidate_predicate,
            )
            msg = resp_msg
            accumulated_usage.pop("_last_llm_error", None)
            accumulated_usage.pop("_last_llm_error_kind", None)
            accumulated_usage.pop("_last_llm_retry_same_request", None)
            accumulated_usage.pop("_last_llm_status_code", None)
            accumulated_usage.pop("_last_llm_provider_code", None)

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
                    payload={
                        "message": msg,
                        "usage": usage,
                    },
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
                usage,
                cost,
                category,
                provider=provider,
                source="loop",
                cost_estimated=cost_estimated,
            )

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if not tool_calls and (not content or not content.strip()):
                event_type, is_provider_glitch, permanent_body_error = _record_and_emit_empty_response(
                    usage=usage, msg=msg, accumulated_usage=accumulated_usage, event_queue=event_queue,
                    drive_logs=drive_logs, task_id=task_id, execution_id=execution_id, round_id=round_id,
                    llm_call_id=llm_call_id, round_idx=round_idx, attempt=attempt, model=model,
                    task_type=task_type, content=content, tool_calls=tool_calls,
                    request_ref=request_ref, response_ref=response_ref, transient_budget=transient_budget,
                    context_fit_event_fields=context_fit_event_fields,
                )
                if event_type == "provider_incomplete_response" and not usage.get("provider_error"):
                    response_cache_bypass_requested = True
                # Transient response glitches retry the same model; permanent body errors fail fast.
                if not permanent_body_error and attempt < transient_budget - 1:
                    if _sleep_within_deadline(
                        min(2.0 ** attempt, _TRANSIENT_BACKOFF_CAP_SEC), deadline_ts
                    ):
                        continue
                    _emit_retry_deadline_exhausted(
                        drive_logs, task_id=task_id, execution_id=execution_id,
                        round_id=round_id, round_idx=round_idx, attempt=attempt,
                        model=model, error_kind=event_type,
                    )
                return None, cost

            accumulated_usage.pop("execution_status", None)
            accumulated_usage.pop("result_status", None)
            accumulated_usage.pop("reason_code", None)
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
                "attempt": attempt + 1,
                "model": display_model,
                "reasoning_effort": effort,
                "prompt_tokens": _round_event["prompt_tokens"],
                "completion_tokens": _round_event["completion_tokens"],
                "cached_tokens": _round_event["cached_tokens"],
                "cache_write_tokens": _round_event["cache_write_tokens"],
                "prompt_cache_ttl": _round_event["prompt_cache_ttl"],
                "cost_usd": cost,
                "response_kind": "tool_calls" if tool_calls else "message",
                "tool_call_count": len(tool_calls),
                "has_text": bool(content and str(content).strip()),
            })
            append_jsonl(drive_logs / "events.jsonl", _round_event)
            return msg, cost

        except UsageAccountingError:
            raise  # Monetary/ledger rails are not provider failures.
        except Exception as e:
            if _record_llm_call_error(
                e,
                _LlmErrorContext(
                    task_id=task_id,
                    task_type=task_type,
                    execution_id=execution_id,
                    round_id=round_id,
                    llm_call_id=llm_call_id,
                    round_idx=round_idx,
                    attempt=attempt,
                    model=model,
                    request_ref=request_ref,
                    drive_logs=drive_logs,
                    event_queue=event_queue,
                    accumulated_usage=accumulated_usage,
                    context_fit_event_fields=context_fit_event_fields,
                ),
            ):
                break
            error_kind = str(accumulated_usage.get("_last_llm_error_kind") or "")
            is_transient = error_kind in _TRANSIENT_RETRY_KINDS
            # Non-transient retryable classes keep the caller's max_retries, but never
            # exceed the loop ceiling (transient_budget) — so an attempt_cap'd fallback
            # candidate does not waste a backoff sleep on an iteration the loop won't run.
            # For the primary, transient_budget >= max_retries, so this is a no-op there.
            attempt_budget = transient_budget if is_transient else min(max_retries, transient_budget)
            if attempt >= attempt_budget - 1:
                break
            backoff = _retry_backoff_sec(accumulated_usage, error_kind, attempt, is_transient)
            if not _sleep_within_deadline(backoff, deadline_ts):
                _emit_retry_deadline_exhausted(
                    drive_logs, task_id=task_id, execution_id=execution_id,
                    round_id=round_id, round_idx=round_idx, attempt=attempt,
                    model=model, error_kind=error_kind,
                )
                break

    return None, 0.0
