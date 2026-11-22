"""Main-loop provider calls: request observation, retries, pricing and usage events."""

from __future__ import annotations

import contextlib
import hashlib
import pathlib
import queue
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros import model_concurrency, working_checkpoint
from ouroboros.llm_observability import (persist_main_response, persist_observed_call, host_step, round_receipt,
                                         emit_main_call_state as _emit_main_llm_call_state)
from ouroboros.context_timing import timed
from ouroboros.config import get_finalization_grace_sec, runtime_setting, NETWORK_WAIT_BACKOFF_MAX_SEC  # noqa: F401 - legacy monkeypatch seam
from ouroboros.deadline_utils import (
    owner_deadline_exhausted,
    main_transport_timeout_sec as _main_transport_timeout,
)
from ouroboros.knowledge import observed_route_stamp
from ouroboros.llm import LLMClient, LocalContextTooLargeError, add_usage
from ouroboros.llm_claudexor import (
    ClaudexorModelError, ClaudexorModelNotDispatched, ClaudexorRefusalError, daemon_not_delivered,
    presence_refusal_unstarted, propagate_model_error, RESTORED_END_KEY, RESTORED_PROCESSING_KEY, restored_burst, restored_round, restored_view,
)
from ouroboros.llm_substitution import same_route_refusal, stamp_substitutions
from ouroboros.model_wait import (
    OWNER_WORDS_WAITING, ModelWaitInterrupted, current_model_wait, model_wait_reason, propagate_model_control,
)
from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY, pop_custom_validation_receipts
from ouroboros.llm_attempt import PROVIDER_POLICY_REFUSAL, _is_provider_policy_refusal, is_size_refusal, single_physical_send, track_physical_attempts
from ouroboros.observability import mark_last_answer, new_call_id, new_execution_id, persist_call
from ouroboros.pricing import emit_llm_usage_event, infer_model_category, normalize_usage_cost as _normalize_usage_cost
from ouroboros.send_clock import main_send_scope
from ouroboros.task_pacing import main_loop_wire_options, task_chat_conversation_id
from ouroboros.task_custody import canonical_owner_root
from ouroboros.transport_custody import attempt_custody_event_fields, attempt_end_known, is_pre_dispatch_transport_failure, outcome_unknown_on_chain
from ouroboros.transport_custody import ObservedEndpointChanged
from ouroboros._usage_response import (CONTENT_REFUSAL_FINISH_REASONS, capture_finish_reason, finish_event_fields,
                                       INTERMEDIATE_FINISH_REASONS, OUTPUT_LIMIT_FINISH_REASONS,
                                       TOOL_CALL_FINISH_REASONS, provider_answer, reply_usability,
                                       assistant_answer_text, refusal_content_text, response_finish_reason, typed_values, usage_reply_usability)
from ouroboros.usage_accounting import PhysicalAttemptContext, PhysicalAttemptPreconditionFailed, UsageAccountingError, bind_physical_attempt_context, current_usage_scope, last_physical_attempt_capture, physical_attempt_capture_from_exception
from ouroboros.physical_attempt_context import observe_dispatch
from ouroboros.primary_route_observation import note_content_refusal
from ouroboros.transcript_prefix import note_prefix_answer, refresh_attempt_prefix, served_account_fields, settle_rewrite_reread
from ouroboros.contracts.record_contract import RECORD_CONTRACT_VERSION
from ouroboros.event_usage import REPORTED_TOKENS_KEY, event_token_counts
from ouroboros.utils import (
    append_jsonl,
    emit_cognitive_operation_event,
    emit_log_event,
    has_log_sink,
    sanitize_tool_result_for_log,
    truncate_review_artifact,
    utc_now_iso,
)

log = logging.getLogger(__name__)

MAIN_LOOP_MAX_TOKENS = 65_536


# Preserve native retrieval evidence across calls; add_usage only folds numbers.
# Counts and capped URLs inform review, without retaining titles or snippets here.
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

# Per-class retry policy: TRANSIENT provider failures (finish_reason=null glitches, 429/5xx/overloaded)
# get a larger same-model attempt budget than permanent classes, because the owner deliberately runs
# single-model setups (all slots = one model, empty fallback) for clean measurement and a 3-attempt cap
# turned recoverable provider blips into whole-task "No viable fallback model configured" deaths.
# Permanent classes (auth/quota/bad_request/too large) keep failing fast. There is NO cross-model
# fallback here — the same request is retried on the SAME model. One narrower class (owner decision
# 21.10.2026, #1886): a configured API child whose stream the provider ended with an explicit timeout
# after model output gets at most ONE repeat (``TIMEOUT_REPEAT_KEY``); after it nothing more is sent on
# the round and its parent, which chose that actor, decides.
_TRANSIENT_RETRY_KINDS = frozenset({"provider_transient", "provider_incomplete_response"})
# OB-01: stamped when THIS invocation spent its same-model retry wall without a usable response; entry-cleared
# per invocation; PERMANENT classes leave it unspent. Its twin: a series of 2+ attempts (not a deadline) ended it.
RETRY_WALL_EXHAUSTED_KEY, RETRY_ATTEMPTS_SPENT_KEY = "_llm_retry_wall_exhausted", "_llm_retry_attempts_spent"
# Error kinds that put a model on the F1 fallback cooldown. Superset of the same-model
# retry kinds: a body-error 429 (HTTP 200 with an error in the body — the canonical
# cloud.ru/OpenRouter rate-limit shape) is classified "rate_limit", which must cool the
# model down even though it is not a same-model retry kind. Kept separate so widening the
# cooldown trigger never enlarges the same-model transient-retry budget.
_COOLDOWN_ERROR_KINDS = _TRANSIENT_RETRY_KINDS | frozenset({"rate_limit"})
# A subscription window that is spent but heals on a timer. SSOT for the name.
# Deliberately NOT `quota_exhausted`: that class is classified PERMANENT, which is
# correct for a billing refusal (402, no credits) and wrong for a plan window whose
# only cure is waiting. Its `reset_at` is a disclosed fact, never a sleep: configured
# routes answer first, then model_wait's visible wait (or inline Presence's typed refusal).
SUBSCRIPTION_WINDOW_EXHAUSTED = "subscription_window_exhausted"
_TRANSIENT_RETRY_DEFAULT = 6
_TRANSIENT_BACKOFF_CAP_SEC = NETWORK_WAIT_BACKOFF_MAX_SEC
# Stop retrying when the remaining task deadline cannot absorb the backoff
# sleep plus a useful follow-up attempt — burning the last minutes of a task
# deadline sleeping between retries is worse than failing visibly.
_DEADLINE_RETRY_FLOOR_SEC = 10.0
# Historical checkpoints may name earlier unresolved attempts only by round and count. Read the
# fence until a usable reply or the next round retires it. New work never writes a repeat grant.
TRANSPORT_DEATHS_KEY = "_transport_deaths"
# Stamped by the round gate before each round call: an open wait episode paces its own redials, so its call sends once.
TRANSPORT_WAIT_REDIAL_KEY = "_transport_wait_redial"
# A local pre-dispatch refusal's own candidate facts (``LocalContextTooLargeError.refused_candidate``), for the
# round's one strict-shrink retry; a wait's reprepare rebinds the physical context of the remaining attempts.
REFUSED_CANDIDATE_KEY, REBOUND_PHYSICAL_CONTEXT_KEY = "_refused_candidate", "_physical_context_rebound"
# The logical round's first started frame ({"round_id", "at": monotonic}); retries, fallbacks and redials keep it.
ROUND_STARTED_KEY = "_llm_round_started"
# A round's timeout repeat ({"round_id", "first": the timed-out attempt's failure row, "budget_spent"}), written when
# it is planned or refused: while it stands the round makes no call but that repeat; only a usable response clears it.
TIMEOUT_REPEAT_KEY = "_llm_timeout_repeat"


def _save_timeout_repeat(usage: Dict[str, Any], *, failed: bool = False) -> None:
    record = usage.get(TIMEOUT_REPEAT_KEY)
    if record is not None:
        if failed:
            record.update(state="spent", repeat={**record.get("repeat", {}), **(usage.get("_last_llm_failure") or {})})
            receipt = last_physical_attempt_capture()
            if "attempt_custody_state" not in (usage.get("_last_llm_failure") or {}):
                record["repeat"]["attempt_custody_state"] = (receipt.state if receipt is not None
                    and receipt.attempt_id == record["repeat"].get("physical_attempt_id") else None)
        working_checkpoint.save_model_call()


def timeout_repeat_blocked(usage: Dict[str, Any]) -> bool:
    """A restored dispatch has no answer receipt; neither a new ID nor round renews it."""
    record = usage.get(TIMEOUT_REPEAT_KEY) or {}
    if record.get("state") not in {"dispatched", "spent"}:
        return False
    failure = record.get("repeat") or {}
    if record["state"] == "dispatched":
        failure = {**failure, "failure_code": "provider_outcome_unknown",
                   "error": "Repeat dispatch was recorded; its provider outcome is UNKNOWN."}
        record["repeat"] = failure
    usage.update(execution_status="infra_failed", reason_code="provider_unavailable",
                 _last_llm_failure=failure, _last_llm_error_kind=failure.get("failure_code", "provider_unavailable"),
                 _last_llm_error=failure.get("error", ""))
    return True


def _claim_timeout_repeat(usage: Dict[str, Any], call_ids: Dict[str, Any], reservation: Any) -> None:
    record = usage[TIMEOUT_REPEAT_KEY]
    if record.get("state") != "granted":
        raise PhysicalAttemptPreconditionFailed("timeout repeat already dispatched")
    record.update(state="dispatched", repeat={**call_ids, "physical_attempt_id": reservation.attempt_id,
                                              "attempt_custody_state": "reserved"})  # write-ahead, not a wire receipt
    _save_timeout_repeat(usage)


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
    raw = runtime_setting("OUROBOROS_TRANSIENT_RETRY_MAX", "").strip()
    try:
        value = int(raw) if raw else default_value
    except ValueError:
        value = default_value
    return max(int(default_retries), value)


def _empty_response_log_msg(usage: Dict[str, Any], is_provider_glitch: bool, exhausted: bool, finish_fields: dict) -> str:
    """Honest message for an empty/incomplete LLM response: a transient provider
    body-error (OpenRouter 429/5xx inside an HTTP 200, surfaced as usage
    ``provider_error``) that a same-model reroute could not escape is named as
    itself, not as a blank finish_reason=null glitch, and an output-exhausted reply
    as itself. Pure: the error KIND is owned by the caller's typed assignment."""
    provider_error = usage.get("provider_error") if isinstance(usage, dict) else None
    if isinstance(provider_error, dict):
        return f"Provider returned a body error (code={provider_error.get('code')}): {provider_error.get('message')}"
    if isinstance(usage, dict) and usage.get("response_status") == "incomplete":
        label = finish_fields.get("finish_reason")
        detail = f" (finish_reason={label})" if label is not None else ""
        return f"LLM reply ended incomplete{detail}; its output is no result"
    if is_provider_glitch:
        detail = " (finish_reason=null)" if finish_fields == {"finish_reason": None} else ""
        return "Provider returned incomplete response" + detail
    if exhausted:
        return "LLM reply ended on its output limit before any visible text or tool call"
    return "LLM returned empty response (no content, no tool_calls)"


def _classify_empty_response(usage: Dict[str, Any], msg: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Classify an empty / no-tool-call response → (event_type, is_provider_glitch,
    permanent), reading the finish once (``response_finish_reason``). Structured
    overflow and a TYPED non-transient body error (WA1 kind ``provider_error``: auth /
    quota / bad_request) are PERMANENT — a same-model reroute already failed in the
    transport, so retrying here only burns the transient budget. So is output
    exhaustion, an empty reply without a body error that ended on its output limit:
    the same request ends the same way. It stays an ``llm_empty_response`` (the only
    permanent one; its kind is ``llm_output_exhausted``). So is the provider's own
    content refusal: ``provider_content_refusal``, kind ``content_filter``, a route fact.
    Only rate_limit / provider_transient body errors and a missing or null finish (a glitch) retry."""
    finish = str(response_finish_reason(usage, msg)[1] or "").strip().lower()
    if finish in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
        return "remote_context_overflow", False, True
    body_err = usage.get("provider_error") if isinstance(usage, dict) else None
    # A refusal under its body's own status; a typed cause of another kind keeps its kind, as on an exception.
    typed_other = isinstance(body_err, dict) and any(
        _provider_code_kind(str(body_err.get(key) or "")) in ("quota_exhausted", "auth_error") for key in ("type", "code"))
    if isinstance(body_err, dict) and not typed_other and (
            (body_err.get("answer") or {}).get("basis") or is_size_refusal(body_err)):
        return "remote_context_overflow", False, True
    if not body_err and finish in OUTPUT_LIMIT_FINISH_REASONS:
        return "llm_empty_response", False, True
    if not body_err and finish in CONTENT_REFUSAL_FINISH_REASONS:
        return "provider_content_refusal", False, True
    is_provider_glitch = not finish
    transient_body = isinstance(body_err, dict) and body_err.get("kind") in ("rate_limit", "provider_transient")
    permanent = bool(body_err) and not transient_body
    event_type = "provider_body_error" if permanent else "provider_incomplete_response" if is_provider_glitch else "llm_empty_response"
    return event_type, is_provider_glitch, permanent


def _attempt_loop_budget(max_retries: int, attempt_cap: Optional[int]) -> int:
    """Attempt-loop ceiling. Normally ``transient_retry_max(max_retries)``; when
    ``attempt_cap`` is set (F1 fallback candidate), cap the WHOLE loop (every error class)
    to a small total so the chain tries a candidate a fixed couple of times then moves on.
    Applied only to candidates; the primary passes None and keeps its full budgets."""
    budget = transient_retry_max(max_retries)
    return budget if attempt_cap is None else max(1, min(int(budget), int(attempt_cap)))


def _record_and_emit_empty_response(
    *, usage, msg, accumulated_usage, event_queue, drive_logs, task_id, execution_id,
    round_id, llm_call_id, round_idx, attempt, model, task_type, content, tool_calls,
    request_ref, response_ref, transient_budget, context_fit_event_fields,
    finish_fields, task_attempt=None, prior_attempt_id=None,
) -> tuple:
    """Classify an empty / no-tool-call response, log + emit its events, and stamp
    accumulated_usage (last error / execution_status / reason_code / F1 cooldown kind).
    Returns ``(event_type, is_provider_glitch, permanent)`` for the caller's retry
    decision. Extracted from call_llm_with_retry to keep that loop readable."""
    body_error = usage.get("provider_error") if isinstance(usage, dict) and isinstance(usage.get("provider_error"), dict) else {}
    event_type, is_provider_glitch, permanent = _classify_empty_response(usage, msg)
    exhausted = event_type == "llm_empty_response" and permanent  # the classifier's output exhaustion
    refused = event_type == "provider_content_refusal"
    words = note_content_refusal(accumulated_usage, usage, msg) if refused else {}  # its own words, for its round's row
    label = finish_fields.get("finish_reason") or "unknown"
    log_msg = (f"The provider's content filter stopped this request (finish_reason={label})" if refused
               else _empty_response_log_msg(usage, is_provider_glitch, exhausted, finish_fields))
    log.warning("%s, attempt %d/%d", log_msg, attempt + 1, transient_budget)
    _emit_empty_response_events(
        event_type, event_queue=event_queue, drive_logs=drive_logs,
        base={"task_id": task_id, "execution_id": execution_id, "round_id": round_id, **_scope_lineage(),
              "llm_call_id": llm_call_id, "round": round_idx, "attempt": attempt + 1,
              "model": model, **finish_fields, "task_attempt": task_attempt,
              # WHICH upstream endpoint served (or refused) this round, plus the typed class
              # of an HTTP-200 body error (the shape issue #468 died on); a durable record
              # naming only the MODEL cannot attribute a same-model provider incident.
              "response_provider": usage.get("response_provider"),
              "provider_error_kind": body_error.get("kind"),
              **words, **context_fit_event_fields},
        task_type=task_type,
        details={"content": content, "tool_calls": tool_calls,
                 "request_ref": request_ref, "response_ref": response_ref},
    )
    if exhausted:  # this attempt's own facts for the next round's host fact (loop.run_llm_loop)
        from ouroboros.loop_output_exhaustion import record_empty_cut

        capture = _own_capture(last_physical_attempt_capture(), prior_attempt_id)
        if (ids := usage.get("ledger_attempt_ids")) and capture is not None and capture.attempt_id != ids[-1]:
            capture = None
        sent = capture.max_completion_tokens if capture is not None and capture.state in {"settled", "dispatched"} else None
        record_empty_cut(accumulated_usage, usage, sent_allowance=sent, message=_short_error_text(log_msg))
    else:
        if event_type == "remote_context_overflow":
            status, reason, kind = "infra_failed", "llm_api_error", "context_overflow"
        else:
            status = "infra_failed" if (is_provider_glitch and not permanent) else "failed"
            reason = event_type
            # Cooldown signal for the F1 fallback gate (see helper; not a retry change).
            kind = "content_filter" if refused else _cooldown_kind_for_empty_response(body_error, event_type)
        error = _short_error_text(log_msg)
        accumulated_usage.update(_last_llm_error=error, execution_status=status, reason_code=reason,
                                 _last_llm_error_kind=kind, _last_llm_attempt_custody="")  # a response arrived
        failure = accumulated_usage["_last_llm_failure"] = accumulated_usage.get("_last_llm_call_meta", {})
        failure.update(failure_code=kind, error=error)
    from ouroboros.loop_transport import stamp_owner_provider_message
    stamp_owner_provider_message(accumulated_usage, {"message": words.get("provider_message")} if refused else body_error)  # its own sentence
    _save_timeout_repeat(accumulated_usage, failed=True)
    return event_type, is_provider_glitch, permanent


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
        if _output_or_body_size_message(str(body_error.get("message") or "")):
            return "request_too_large"
    return body_kind if body_kind in _COOLDOWN_ERROR_KINDS else event_type


def _retry_backoff_sec(attempt: int, is_transient: bool) -> float:
    """Seconds before retrying the same request; no reset instant ever schedules one."""
    return min(2.0 ** attempt * 4, _TRANSIENT_BACKOFF_CAP_SEC if is_transient else 30.0)


def _sleep_within_deadline(
    seconds: float, deadline_ts: Optional[float], *, wake_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Sleep ``seconds`` if the task deadline (epoch seconds) allows another
    attempt afterwards. Returns False — without sleeping — when the remaining
    time budget cannot absorb the backoff, signalling the caller to stop.
    An optional wake also returns False after interrupting sleep: the paid
    transport repeat's typed control, an ordinary repeat's unread owner words."""
    if deadline_ts and float(deadline_ts) - time.time() < float(seconds) + _DEADLINE_RETRY_FLOOR_SEC:
        return False
    if wake_check is not None:
        from ouroboros.loop_transport import interruptible_wait_sleep

        return not interruptible_wait_sleep(seconds, wake_check)
    time.sleep(float(seconds))
    return True


def _emit_retry_deadline_exhausted(
    drive_logs: pathlib.Path, *, task_id: str, execution_id: str, round_id: str,
    round_idx: int, attempt: int, model: str, error_kind: str,
) -> None:
    """Durable record that retries stopped because the deadline could not
    absorb another backoff sleep (emitted by BOTH transient failure paths)."""
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(), "type": "llm_retry_deadline_exhausted",
        "task_id": task_id, "execution_id": execution_id, "round_id": round_id,
        "round": round_idx, "attempt": attempt + 1,
        "model": model, "error_kind": error_kind,
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
    if accumulated_usage.get(TRANSPORT_DEATHS_KEY) and accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown":
        accumulated_usage["_last_llm_retry_same_request"] = False
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
    owner_input_check: Optional[Callable[[], List[str]]] = None  # unread owner input ids: a granted repeat yields to them
    requested_profile: Optional[str] = None
    use_local: Optional[bool] = None  # the asked route's locality, for its failure row
    configured_api_child: bool = False  # its parent chose this exact API actor (``TIMEOUT_REPEAT_KEY``)
    prior_attempt_id: Optional[str] = None  # the attempt last captured before this call began (``_own_capture``)


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
    # A size refusal's recognition basis (``context_budget.size_refusal``) and the ``(speaker, sentence)`` that carried it.
    basis: str = ""
    said: Tuple[str, str] = ("", "")
    # A client refusal no typed code explains: only such an answer may be recognised by a cited declared limit.
    untyped: bool = False


def _emit_live_log(event_queue: Optional[queue.Queue], payload: Dict[str, Any]) -> None:
    """Thin wrapper around the SSOT helper — keeps the call-site signature stable."""
    timed(
        "event_put", emit_log_event, event_queue,
        {"ts": utc_now_iso(), **payload},
        log_label="LLM live",
    )


def _short_error_text(value: Any, limit: int = 220) -> str:
    text = " ".join(sanitize_tool_result_for_log(str(value or "")).split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


from ouroboros.context_budget import (  # one overflow vocabulary for every seam
    CONTEXT_OVERFLOW_CODES as _STRUCTURED_CONTEXT_OVERFLOW_CODES,
    OUTPUT_OR_BODY_SIZE_MARKERS, SIZE_REFUSAL_STATUSES,
    output_or_body_size_message as _output_or_body_size_message, size_refusal,
)
_FORCED_INCOMPLETE_FINISH_REASONS = (_STRUCTURED_CONTEXT_OVERFLOW_CODES | OUTPUT_LIMIT_FINISH_REASONS
                                     | TOOL_CALL_FINISH_REASONS | INTERMEDIATE_FINISH_REASONS)


def forced_response_is_incomplete(response_meta: Optional[Dict[str, Any]], *, completion_request: bool = False) -> bool:
    """Return whether a forced provider response is not a complete final. ``completion_request`` waives
    exactly the reply's one completion call (its count and tool-call finish) and a provider length cut
    (owner 30 Oct, A: a whole request is the author's act; its own parse refuses a cut inside it, and
    ``forced_response_cut`` keeps the fact); an overflowed or null-finished response stays incomplete.
    The response facts are read, never rewritten."""
    if not isinstance(response_meta, dict):
        return False
    finish_reason = response_meta.get("finish_reason")
    reasons = _FORCED_INCOMPLETE_FINISH_REASONS - (
        TOOL_CALL_FINISH_REASONS | OUTPUT_LIMIT_FINISH_REASONS if completion_request else frozenset())
    if reply_usability(finish_reason, response_meta.get("response_status"), {}) == "unusable":
        return True
    return int(response_meta.get("tool_call_count") or 0) > int(completion_request) or (
        response_meta.get("finish_reason_present") is True
        and (
            finish_reason is None
            or str(finish_reason).strip().lower() in reasons
        )
    )


def forced_response_cut(response_meta: Optional[Dict[str, Any]]) -> str:
    """The provider's output-limit finish of a forced response, '' when it named none."""
    meta = response_meta if isinstance(response_meta, dict) else {}
    finish = str(meta.get("finish_reason") or "").strip().lower()
    return finish if meta.get("finish_reason_present") is True and finish in OUTPUT_LIMIT_FINISH_REASONS else ""


_NON_RETRYABLE_PROVIDER_MARKERS = {
    "quota_exhausted": (
        "insufficient credits",
        "insufficient_credit",
        "insufficient_quota",
        "1113",  # Z.ai "Insufficient balance" (HTTP 429): billing, not a rate limit
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
    "request_too_large": (*OUTPUT_OR_BODY_SIZE_MARKERS, "too many tokens"),
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
    "tokens per minute",
    "requests per minute",
    "token per minute",
    "request per minute",
)
# Short markers count only as whole tokens, so "1004290 tokens" or "req_1f429ab0" is not a rate limit.
_RATE_LIMIT_TEXT_PATTERN = re.compile(r"(?<![a-z0-9])(?:429|tpm|rpm)(?![a-z0-9])")
# The second is OpenRouter's 402 for money held by requests still in flight (``error.metadata.limit_source``, its limits page of 13 Nov 2026): a pause.
_RETRYABLE_PROVIDER_CODES = frozenset({"rate_limit_exceeded", "openrouter_in_flight_budget"})


def _is_rate_limit_text(text: str) -> bool:
    low = str(text or "").lower()
    return any(marker in low for marker in _RATE_LIMIT_TEXT_MARKERS) or bool(_RATE_LIMIT_TEXT_PATTERN.search(low))


is_rate_limit_text = _is_rate_limit_text  # public SSOT aliases for the safety lane
NON_RETRYABLE_PROVIDER_MARKERS = _NON_RETRYABLE_PROVIDER_MARKERS


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


def _exception_provider_code(values: List[str], body_values: List[str], size_family: bool) -> str:
    """The failure's leading typed value: a size code inside the client family, a retryable code, else the first
    named code the classifier types."""
    generic_bad_request_wrappers = {
        "badrequest",
        "badrequesterror",
        "invalidrequest",
        "invalidrequesterror",
        "error",
    }

    def generic(value: str) -> bool:
        return value == "400" or value.lower().replace("_", "").replace("-", "").replace(" ", "") in generic_bad_request_wrappers

    for wanted in ((_STRUCTURED_CONTEXT_OVERFLOW_CODES,) if size_family else ()) + (_RETRYABLE_PROVIDER_CODES,):
        for value in values:
            if value.lower() in wanted:
                return value
    for value in values:
        if not value.isdigit() and not generic(value) and _provider_code_kind(value):
            return value
    # A response body that repeats only the generic numeric 400 is not a
    # provider-specific code. Leave it empty so its message can classify the
    # actual context/output failure; attr-only numeric codes stay available.
    if body_values and all(generic(value) for value in body_values):
        return ""
    return values[0] if values else ""


def _exception_provider_message(exc: Exception, safe_error: str = "") -> str:
    """The Logs diagnostic of a failure: the innermost party's own sentence (``provider_answer``), sanitized and
    truncated, else the exception's printed form. Two refusals can share a status and code (a content-filter block,
    a rejected field), so the durable event keeps the words; never the owner row or an input to classification."""
    engine = getattr(exc, "problem", None)  # an engine-typed problem's own message is its Logs diagnostic
    return sanitize_tool_result_for_log(provider_answer(exc).innermost()[1] or str(
        (engine.get("message") if isinstance(engine, dict) else "") or safe_error or "").strip())[:600]


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
    if isinstance(exc, ClaudexorRefusalError):
        return LlmErrorClassification(exc.code, False, provider_code=exc.code)
    if getattr(exc, "stream_rejected", False): return LlmErrorClassification("provider_error", False)  # complete wire judged unusable: a local verdict, never a same-model repeat
    if getattr(exc, "code", "") == "model_outcome_unknown":
        return LlmErrorClassification("provider_outcome_unknown", False)
    if getattr(exc, "code", "") == "model_substituted": return LlmErrorClassification("model_substituted", False, 0, exc.code)  # a well-formed request the route answered with another model: chain-eligible, never cools the requested one
    if getattr(exc, "code", "") in {"unsupported_parameter", "invalid_continuation", "auth_changed", "model_unavailable", "model_source_unavailable", "model_catalog_identity_mismatch", "model_capability_refused", "model_retry_invalid"}:
        return LlmErrorClassification("bad_request", False, _exception_status_code(exc), exc.code)
    if getattr(exc, "code", "") in {"auth_required", "auth_unavailable", "auth_refresh_failed"}:
        return LlmErrorClassification("auth_error", False, _exception_status_code(exc), exc.code)
    if isinstance(exc, LocalContextTooLargeError):
        return LlmErrorClassification("context_overflow", False)
    # A reset forecast is evidence, not proof that an unavailable pool exhausted quota.
    wcode, reset_at = str(getattr(exc, "code", "") or ""), str(getattr(exc, "reset_at", "") or "")
    if wcode == SUBSCRIPTION_WINDOW_EXHAUSTED:
        return LlmErrorClassification(wcode, False, _exception_status_code(exc), reset_at=reset_at)
    if wcode == "credential_pool_exhausted" and reset_at:
        return LlmErrorClassification("provider_error", False, _exception_status_code(exc), wcode, reset_at=reset_at)
    status_code = _exception_status_code(exc)
    # The final status sets the family: only a client refusal, or an answer with no status, is read for size.
    size_family = not status_code or status_code in SIZE_REFUSAL_STATUSES
    answer = provider_answer(exc)
    values = typed_values(exc, answer)
    provider_code = _exception_provider_code(values, answer.own_codes(), size_family)
    # Typed refusal (llm_attempt.ProviderPolicyRefusal): nothing upstream answered,
    # permanent by class — structural, and it outranks every prose heuristic below.
    if _is_provider_policy_refusal(exc):
        return LlmErrorClassification(PROVIDER_POLICY_REFUSAL, False, status_code, provider_code or PROVIDER_POLICY_REFUSAL)
    # Its current path's own words, else its printed form: an earlier upstream attempt never re-types the final answer.
    low = "\n".join([text for _speaker, text, history in answer.said if not history] or [safe]).lower()
    if size_family and provider_code.lower() in _STRUCTURED_CONTEXT_OVERFLOW_CODES:
        return LlmErrorClassification("context_overflow", False, status_code, provider_code, basis="typed",
                                      said=(size_refusal(answer, status_code, [provider_code]) or (*[""] * 2, *answer.innermost()))[2:])
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
    if status_code == 429 or (not status_code and _is_rate_limit_text(low)):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if size := size_refusal(answer, status_code, printed=safe):  # size evidence outranks output words on the same answer
        return LlmErrorClassification("context_overflow", False, status_code, size[1] or provider_code, basis=size[0], said=size[2:])
    untyped = status_code in SIZE_REFUSAL_STATUSES
    if size_family and any(_output_or_body_size_message(text) for text in (*(row[1] for row in answer.said), safe)):
        return LlmErrorClassification("request_too_large", False, status_code, provider_code, untyped=untyped)
    if provider_kind:
        return LlmErrorClassification(provider_kind, False, status_code, provider_code, untyped=untyped)
    for kind, markers in _NON_RETRYABLE_PROVIDER_MARKERS.items():
        # A known final status is the status: digits inside a sentence (a count like 4026546) never restate it.
        if any(marker in low for marker in markers if not (status_code and marker.isdigit())):
            return LlmErrorClassification(kind, False, status_code, provider_code, untyped=untyped)
    if status_code in {400, 401, 402, 403, 422}:
        kind = {
            400: "bad_request",
            401: "auth_error",
            402: "quota_exhausted",
            403: "auth_error",
            422: "bad_request",
        }[status_code]
        return LlmErrorClassification(kind, False, status_code, provider_code, untyped=untyped)
    if status_code == 408 or (status_code is not None and 500 <= status_code <= 599):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    if status_code is not None and 400 <= status_code <= 499:
        return LlmErrorClassification("provider_error", False, status_code, provider_code)
    capture = getattr(exc, "physical_attempt_capture", None)
    if outcome_unknown_on_chain(exc):
        # The provider may still finish a request whose socket outcome is
        # unknown, even when an earlier usage frame settled its exact price.
        # Money does not establish a terminal response. Typed provider errors
        # above retain their existing classifications and retry policy.
        return LlmErrorClassification(
            "provider_outcome_unknown", False, status_code, provider_code,
        )
    if (
        str(getattr(capture, "state", "") or "") == "released"
        and str(getattr(capture, "provider", "") or "") != "local"
        and not bool(getattr(capture, "route_is_loopback", False))
        and is_pre_dispatch_transport_failure(exc) and not model_wait_reason(exc)  # an engine resource verdict is no outage
    ):
        # Typed $0 fact: the request never reached a REMOTE provider (released custody + typed pre-dispatch
        # failure). Retrying the same request is free and safe; the engine's not-delivered receipt repeats within
        # the ordinary bound (``_stop_after_llm_error``), other pacing is the round-level episode's in loop.py. A local
        # provider's connect failure stays on the generic path — a stopped local server is not a network outage worth
        # waiting out — and so does a loopback OpenAI-compatible route (Ollama / LM Studio / vLLM): its provider
        # stamp is remote-shaped but the server is on this host.
        return LlmErrorClassification(
            "transport_unavailable", True, status_code, provider_code,
        )
    if any(marker in low for marker in _RETRYABLE_PROVIDER_MARKERS):
        return LlmErrorClassification("provider_transient", True, status_code, provider_code)
    return LlmErrorClassification("provider_error", True, status_code, provider_code)


def confirmed_api_resource_refusal(exc: Exception) -> str:
    """A received API response about this route's key or bill. Its price can still be unknown."""
    from ouroboros._usage_response import provider_failure_payload
    from ouroboros.transport_custody import attempt_end_known, stream_incomplete_on_chain

    capture = getattr(exc, "physical_attempt_capture", None)
    status = getattr(capture, "provider_status_code", None)
    provider = getattr(capture, "provider", "")
    response = getattr(exc, "response", None)
    http_status = getattr(response, "status_code", None)
    state = getattr(capture, "state", None)
    if (status not in (400, 401, 402, 403, 429) or status != http_status or not getattr(capture, "attempt_id", "")
            or state not in {"settled", "unresolved"} or not provider or provider in {"claudexor", "local"}):
        return ""
    if (isinstance(getattr(exc, "problem", None), dict) or _is_provider_policy_refusal(exc)
            or getattr(exc, "stream_rejected", False) or getattr(exc, "stream_receipt", None)
            or getattr(exc, "stream_usage", None) or stream_incomplete_on_chain(exc) or not attempt_end_known(capture, exc)):
        return ""
    payload = provider_failure_payload(exc)
    answer = provider_answer(payload)
    billing_sources = {"openrouter_credits", "openrouter_key_limit"} if provider == "openrouter" and status == 402 else set()
    if any(obj.speaker or obj.history or (obj.parent >= 0 and set(obj.codes) - billing_sources) for obj in answer.objects):
        return ""
    # The physical receipt carries the outer provider's structured fields. A nested error's
    # code cannot establish a billing or credential problem on the route that sent it.
    code = str(getattr(capture, "provider_code", "") or "").lower()
    error_type = str(getattr(capture, "provider_error_type", "") or "").lower()
    causes = {"invalid_api_key": "auth", "authentication_error": "auth", "insufficient_quota": "quota",
              "billing_hard_limit_reached": "quota", "billing_not_active": "quota"}
    if provider == "zai":
        causes["1113"] = "quota"
    generic_types = {"", type(exc).__name__.lower(), "invalid_request_error", "api_error"}
    typed = code if code and code != str(status) else error_type
    if reason := causes.get(typed):
        return reason if error_type in generic_types or causes.get(error_type) == reason else ""
    by_status = {401: "auth", 402: "quota"}.get(status)
    if not by_status or code not in {"", str(status), "unauthorized" if status == 401 else "payment_required"}:
        return ""
    if error_type not in generic_types and error_type != code:
        return ""
    # A bare HTTP refusal, a provider's matching code, or an authentication challenge names the
    # resource. Message-only gateway bodies do not: don't decide what to do by reading their prose.
    if (not payload or code or (billing_sources and billing_sources.intersection(answer.own_codes()))
            or (status == 401 and getattr(response, "headers", {}).get("www-authenticate"))):
        return by_status
    return ""


def model_wait_access_reason(error: Exception) -> str:
    """What this refusal proves for the wait and fallback cards, not permission to wait or retry."""
    if api := confirmed_api_resource_refusal(error):
        return "api_key" if api == "auth" else "api_billing"
    reason = model_wait_reason(error, caller_reason=False)
    if reason == "connection_unavailable":
        return reason
    capture = getattr(error, "physical_attempt_capture", None)
    return reason if isinstance(error, ClaudexorModelError) and getattr(capture, "state", None) in {"released", "settled"} else ""


def _remember_llm_call(
    usage: Dict[str, Any], *, call_ids: Dict[str, Any], model: str, display_model: str, provider: str,
    request_ref: Dict[str, Any], response_ref: Dict[str, Any],
    reported_model: Any = None, use_local: Optional[bool] = None,
    requested_profile: Optional[str] = None, observed_route: Optional[Dict[str, Any]] = None,
) -> None:
    call_meta = {"ts": utc_now_iso(), **call_ids,
        "model": model, "resolved_model": display_model, "provider": provider,
        "reported_model": reported_model.strip() if isinstance(reported_model, str) and reported_model.strip() else None,
        "use_local": use_local,
        "request_ref": request_ref.get("manifest_ref") if request_ref else None,
        "response_ref": response_ref.get("manifest_ref") if response_ref else None,
        "requested_profile": requested_profile, "observed_route": dict(observed_route or {}),
    }
    usage["_last_llm_call_meta"] = call_meta
    usage.setdefault("llm_call_refs", []).append(call_meta)


def _scope_lineage() -> Dict[str, str]:
    """The producer's own known lineage, never a history walk."""
    scope = current_usage_scope()
    return {key: str(getattr(scope, key)) for key in ("root_task_id", "parent_task_id") if getattr(scope, key, "")}


def _record_llm_call_error(
    error: Exception,
    ctx: _LlmErrorContext,
) -> bool:
    """Classify an LLM-round exception, emit live/durable evidence and mark usage infra-failed.
    Return whether the caller must stop retrying the unchanged request."""
    safe_error = sanitize_tool_result_for_log(repr(error))
    classification = classify_llm_exception(error, safe_error)
    if ctx.task_type == "presence":  # stays True only while every attempt so far provably never started
        ctx.accumulated_usage["_presence_pre_dispatch_only"] = bool(ctx.accumulated_usage.get("_presence_pre_dispatch_only", True) and presence_refusal_unstarted(error, ctx.round_idx))
    provider_message = _exception_provider_message(error, safe_error)
    # Display metadata must not enter the classifier's text heuristics.
    display_message = getattr(error, "display_message", None)
    display_error = sanitize_tool_result_for_log(display_message) if isinstance(display_message, str) and display_message else safe_error
    custody_fields = attempt_custody_event_fields(error)
    timed_out = bool(ctx.configured_api_child and custody_fields.get("explicit_timeout") and custody_fields.get("model_output_received"))
    will_retry = classification.retry_same_request or timed_out  # a typed timeout after output, HTTP-shaped or not (#1886)
    legacy = ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY)
    if classification.kind == "provider_outcome_unknown":
        ctx.accumulated_usage["_pending_transport_outcome"] = {
            **custody_fields, "model": ctx.model, "operation_id": str(getattr(error, "operation_id", "") or ""),
            "model_role": str(getattr(error, "model_role", "") or ""),
            "route": dict(getattr(error, "route", {}) or {}), "outcome": "unknown",
            "request_ref": ctx.request_ref.get("manifest_ref") if ctx.request_ref else None,
            **({"same_operation_recoverable": True} if getattr(error, "same_operation_recoverable", False) else {}),
        }
    if isinstance(legacy, dict) and legacy.get("count") and classification.kind != "transport_unavailable":
        will_retry = False
    identity = {
        "task_id": ctx.task_id, "execution_id": ctx.execution_id, "round_id": ctx.round_id, **_scope_lineage(),
        "llm_call_id": ctx.llm_call_id, "round": ctx.round_idx, "attempt": ctx.attempt + 1,
        "model": ctx.model,
    }
    identity, fit, request_ref, restored = restored_view(error, identity, ctx.context_fit_event_fields or {},
                                                         ctx.request_ref.get("manifest_ref") if ctx.request_ref else None)
    # The append sink owns delivery; queue fallback preserves the same identity.
    # No llm_round_error sibling: Background Consciousness owns its separate producer.
    error_event = {
        "ts": utc_now_iso(), "type": "llm_api_error", **identity, "error": display_error,
        "error_kind": classification.kind, "retry_same_request": will_retry,
        "status_code": classification.status_code, "provider_code": classification.provider_code,
        "provider_message": provider_message,
        "requested_profile": ((getattr(error, "usage", {}) or {}).get("claudexor") or {}).get("requested_profile", ctx.requested_profile),
        "observed_route": dict(getattr(error, "route", {}) or {}),
        **custody_fields, **getattr(error, "received_status", {}),
        **({"operation_id": str(error.operation_id)} if getattr(error, "operation_id", "") else {}),
        **fit, "ledger_attempt_ids": list(getattr(error, "ledger_attempt_ids", None) or []), "request_ref": request_ref,
    }
    if not append_jsonl(ctx.drive_logs / "events.jsonl", error_event) or not has_log_sink():
        emit_log_event(ctx.event_queue, error_event, log_label="LLM call error")
    if (owner := canonical_owner_root(ctx.drive_logs.parent)) != ctx.drive_logs.parent.resolve(strict=False):
        append_jsonl(owner / "logs" / "events.jsonl", error_event, live=False)  # REPLICA_RULE: the drive goes, the charge stays
    failure = {"error": _short_error_text(display_error), "resource_refusal": model_wait_reason(error),
        "operation_id": str(getattr(error, "operation_id", "") or ""), **getattr(error, "received_status", {}),
        **restored,  # a restored end's row: that attempt's, a settled one with its own operation (``restored_view``)
        "not_dispatched_provider": (getattr(getattr(error, "physical_attempt_capture", None), "provider", "")
                                    if isinstance(error, ClaudexorModelNotDispatched) else ""),
        **{key: error_event[key] for key in ("ts", "llm_call_id", "round", "model", "requested_profile", "observed_route", "status_code", "provider_code")},
        "use_local": ctx.use_local, "failure_code": classification.kind, "reset_at": classification.reset_at, **{key: value for key, value in custody_fields.items()
            if key in ("physical_attempt_id", "attempt_custody_state", "generation_id", "explicit_timeout", "model_output_received")}}
    if access_reason := model_wait_access_reason(error):
        failure["access_reason"] = access_reason
    if (classification.kind == "bad_request" and (bindings := getattr(error, "refused_option_bindings", ()))
            and attempt_end_known(getattr(error, "physical_attempt_capture", None), error)):
        failure["refused_option_bindings"] = bindings
    if timed_out or ctx.configured_api_child and TIMEOUT_REPEAT_KEY in ctx.accumulated_usage:
        failure["manifest_ref"] = persist_call(ctx.drive_logs.parent, task_id=ctx.task_id,
            call_id=ctx.llm_call_id + "_error", call_type="llm_error", payload=error_event)["manifest_ref"]
    ctx.accumulated_usage.setdefault("llm_call_refs", []).append(failure)
    ctx.accumulated_usage["_last_llm_failure"] = failure  # reuse this attempt's row, not the wait's kind
    ctx.accumulated_usage.update(_last_llm_error=failure["error"], _last_llm_resource_refusal=failure["resource_refusal"],
                                 _last_llm_error_kind=classification.kind, _last_llm_retry_same_request=will_retry, _last_llm_attempt_custody=custody_fields.get("attempt_custody_state", ""))
    if classification.reset_at:  # a disclosed fact for the terminal hint, never a retry schedule
        ctx.accumulated_usage["_last_llm_reset_at"] = classification.reset_at
    else:
        ctx.accumulated_usage.pop("_last_llm_reset_at", None)
    from ouroboros.size_refusal import stamp_failure
    stamp_failure(ctx.accumulated_usage, error, classification)  # the event above keeps the diagnostic provider_message
    for key, value in (("_last_llm_status_code", classification.status_code),
                       ("_last_llm_provider_code", classification.provider_code)):
        if value:
            ctx.accumulated_usage[key] = value
    stamp_substitutions(ctx.accumulated_usage, error)
    ctx.accumulated_usage.update(execution_status="infra_failed", reason_code="llm_api_error")
    if classification.kind == "context_overflow":
        if getattr(error, "refused_candidate", None):  # the local lane's pre-dispatch refusal: its own comparison facts
            ctx.accumulated_usage[REFUSED_CANDIDATE_KEY] = dict(error.refused_candidate)
        overflow_event_type = "local_context_overflow" if isinstance(error, LocalContextTooLargeError) else "remote_context_overflow"
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": overflow_event_type, **identity, "error": safe_error, **fit,
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
    """Retry decision for the exception path: ``True`` stops the attempt loop. Every stop except a one-attempt
    transport stop, an unknown outcome and a granted timeout repeat's own failure stamps the OB-01 marker (a spent
    not-delivered burst does): permanent classes stopped earlier, so a stop here means "the same-model wall is spent"."""
    accumulated_usage = ctx.accumulated_usage
    error_kind = str(accumulated_usage.get("_last_llm_error_kind") or "")
    failure = accumulated_usage.get("_last_llm_failure") or {}
    if TIMEOUT_REPEAT_KEY in accumulated_usage:
        return True  # its repeat failed, whatever the class: the no-call terminal reads the record and this row
    if error_kind == "transport_unavailable" and (
            (failure.get("provider_code") != "transport_not_delivered" and not daemon_not_delivered(failure))
            or accumulated_usage.get(TRANSPORT_WAIT_REDIAL_KEY) or isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)):
        # One free attempt: configured routes and the round's episode own an unreachable route's recovery, an open episode
        # paces its own redials and a held paid repeat sends nothing more. The engine's not-delivered receipt and a typed
        # Claudexor prepare failure before create first repeat within the ordinary bound, as an overloaded route does.
        return True
    if error_kind == "provider_outcome_unknown":
        return True  # only the round owner can admit a new generation after UNKNOWN
    is_transient = error_kind in _TRANSIENT_RETRY_KINDS
    timed_out = ctx.configured_api_child and failure.get("explicit_timeout") and failure.get("model_output_received")
    # Non-transient retryables: max_retries capped by the loop ceiling (primary: no-op); a timeout after output: one.
    attempt_budget = (min(ctx.transient_budget, ctx.attempt + 2) if timed_out else ctx.transient_budget if is_transient
                      else min(ctx.max_retries, ctx.transient_budget))
    backoff = (  # a resource refusal or spent window never resends nor sleeps to a reset
        _retry_backoff_sec(ctx.attempt, is_transient)
        if ctx.attempt < attempt_budget - 1 and not accumulated_usage.get("_last_llm_resource_refusal")
        and error_kind != SUBSCRIPTION_WINDOW_EXHAUSTED else None
    )
    if timed_out:  # recorded before any sleep: a repeat the loop ceiling or the deadline refuses still hands over
        accumulated_usage[TIMEOUT_REPEAT_KEY] = {"round_id": ctx.round_id, "first": dict(failure),
            "state": "granted" if backoff is not None else "spent", "budget_spent": backoff is None}
        _save_timeout_repeat(accumulated_usage)
    if backoff is not None:
        # A granted repeat re-sends the request as built. Owner words that arrived meanwhile end its wait: the round start
        # drains them and rebuilds, same route and round (#2996 joins Main follow-ups to this turn). Words a yield already
        # met cannot wake again (a row that drain leaves unread); a timeout repeat resends its own bytes.
        unread = None if timed_out else ctx.owner_input_check
        fresh = (lambda: not set(unread()) <= set(accumulated_usage.get("_owner_words_met") or ())) if unread else None
        if _sleep_within_deadline(backoff, ctx.deadline_ts, **({"wake_check": fresh} if fresh else {})):
            return False
        if fresh is not None and fresh():  # woken for them, or out of time: the round start reads them first
            accumulated_usage["_owner_words_met"] = [mark for mark in unread() if mark != "direct_incoming"]  # drained
            append_jsonl(ctx.drive_logs / "events.jsonl", {"ts": utc_now_iso(), "type": "llm_not_dispatched", "task_id": ctx.task_id,
                         "round": ctx.round_idx, "model": ctx.model, "reason_code": OWNER_WORDS_WAITING, "error_kind": error_kind})
            raise ModelWaitInterrupted(OWNER_WORDS_WAITING)
        _emit_retry_deadline_exhausted(
            ctx.drive_logs, task_id=ctx.task_id, execution_id=ctx.execution_id,
            round_id=ctx.round_id, round_idx=ctx.round_idx, attempt=ctx.attempt,
            model=ctx.model, error_kind=error_kind,
        )
    accumulated_usage.update({RETRY_WALL_EXHAUSTED_KEY: True, RETRY_ATTEMPTS_SPENT_KEY: 1 < attempt_budget <= ctx.attempt + 1})
    return True


def _empty_response_wall_spent(is_provider_glitch: bool, permanent_body_error: bool, usage: Dict[str, Any]) -> bool:
    """Empty-response exit: spent only while the PROVIDER is failing (finish=None
    glitch or transient body error); a permanent body error and a live provider
    (finish="stop"/"length") both keep their one forced call."""
    return not permanent_body_error and (is_provider_glitch or bool(usage.get("provider_error")))


def provider_no_call_source(accumulated_usage: Dict[str, Any], deadline_exhausted: bool) -> Tuple[str, bool]:
    """The provider-unavailable rail's no-call decision → (typed source, wall_spent).
    Unknown in-flight outcome forbids a RESEND (outranks the wall); a SPENT same-model wall makes
    one more forced call a second full retry window; the deadline_local rail keeps its grace call,
    so ``deadline_exhausted`` suppresses the wall but not a granted timeout repeat (its parent
    decides). A historical transport-death count blocks the resend until a usable response clears it.
    Sources: ``recovered_operation_no_resend`` (a read's own known end or a rejected receipt, on the last failure row or the acting route's),
    ``provider_outcome_unknown_no_resend``, ``resource_refusal_no_resend`` and ``same_route_refusal_no_resend`` (typed
    refusals, a restored Standard repair refused unsent included), ``timeout_repeat_spent_no_resend`` (``TIMEOUT_REPEAT_KEY``),
    ``retry_wall_exhausted_no_repay`` (``RETRY_WALL_EXHAUSTED_KEY``, a per-invocation bool on the shared usage dict). The
    rail stamps ``provider_unavailable`` except for ``resource_refusal_no_resend``."""
    if any((row := accumulated_usage.get(key) or {}).get("recovered_operation_id") or row.get("rejected_receipt") for key in ("_last_llm_failure", "_acting_route_failure")):
        return "recovered_operation_no_resend", False
    if (str(accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown"
            or accumulated_usage.get("_pending_transport_outcome") or isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)):
        return "provider_outcome_unknown_no_resend", False
    if accumulated_usage.get("resource_refusal"): return "resource_refusal_no_resend", True  # typed temporary refusal: no route or owner wait recovered it
    if same_route_refusal(accumulated_usage): return "same_route_refusal_no_resend", False  # a salvage call is the same request on the same route, and the accounts that refused it were just marked
    if TIMEOUT_REPEAT_KEY in accumulated_usage: return "timeout_repeat_spent_no_resend", True  # its repeat answered nothing
    if not deadline_exhausted and bool(accumulated_usage.get(RETRY_WALL_EXHAUSTED_KEY)):
        return "retry_wall_exhausted_no_repay", True
    return "", False


def _emit_empty_response_events(
    event_type: str, *, event_queue: Optional[queue.Queue], drive_logs: pathlib.Path,
    base: Dict[str, Any], task_type: str, details: Dict[str, Any],
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
        "context_raw_input_tokens": int(usage.get("_context_raw_input_tokens") or 0),
        "context_reply_allowance_tokens": int(usage.get("_context_reply_allowance_tokens") or 0),  # planned, not sent
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
    round_id: str = "",
    model: str = "",
    provider: str = "",
) -> tuple:
    """Cache facts of this round only (no dollar counterfactuals).

    Returns ``(prompt_cache_ttl, cache_hit_rate, gap_since_prev_round_sec, duration_ms)``: the
    provider's cached share and the gap since the previous successful round, which separates a
    TTL expiry (long wait) from a prefix rewrite. Also records the APPLIED TTL of this task's
    latest send — the recorded fact the wait-tool cache-horizon disclosure reads (one place, never
    re-derived). ``duration_ms`` spans the logical round from its first started frame (retries,
    fallbacks and waits included) and settles it: a later call on the same round id (a wrap-up)
    measures only itself. No verdict ("cold restart") is derived from these counters: on a
    Codex route the provider reports no cache writes, so one would never fire there.
    """
    reported = event_token_counts(usage)  # #1578: REPORTED counters; an unreported one makes these null
    prompt_tokens, cached_tokens = reported["prompt_tokens"], reported["cached_tokens"]
    prompt_cache_ttl = str(usage.get("prompt_cache_ttl") or "")
    known = None not in (prompt_tokens, cached_tokens)
    cache_hit_rate = (cached_tokens / prompt_tokens if prompt_tokens else 0.0) if known else None
    now, prev_finished = time.monotonic(), accumulated_usage.get("_last_llm_round_finished_monotonic")
    gap_since_prev_round_sec = round(max(0.0, now - float(prev_finished)), 1) if isinstance(prev_finished, (int, float)) else None
    started = accumulated_usage.pop(ROUND_STARTED_KEY, None)
    duration_ms = round(max(0.0, now - float(started["at"])) * 1000) if isinstance(started, dict) and started.get("round_id") == round_id else None
    accumulated_usage["_last_llm_round_finished_monotonic"] = now
    accumulated_usage["_last_prompt_cache_ttl"] = prompt_cache_ttl
    earlier = accumulated_usage.get("_last_round_cache_usage") or {}
    accumulated_usage["_last_round_cache_usage"] = {  # also this accepted call's own money, as its producer stated it
        "round": round_idx, "model": model, "provider": provider, "billing": (usage.get("cost_evidence") or {}).get("billing"),
        **{key: usage.get(key) for key in ("prompt_tokens", "cached_tokens", "cache_write_tokens", "cost", "cost_final")},
        "finished_at": time.time(), **served_account_fields(usage, earlier, provider, model),  # wall clock: survives a restart
        REPORTED_TOKENS_KEY: reported}  # cache routing reads the billing-shaped counters above; route facts these
    settle_rewrite_reread(accumulated_usage)  # this answer measures the history change it carried
    return prompt_cache_ttl, cache_hit_rate, gap_since_prev_round_sec, duration_ms


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
    model_role: str = "main",
    model_account_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from ouroboros import vision_routing

    try:
        return vision_routing.prepare_messages_for_send(
            messages,
            routing=vision_routing.VisionRoutingContext(
                model=model, llm=llm, accumulated_usage=accumulated_usage,
                drive_root=drive_root, task_id=task_id, event_queue=event_queue,
                use_local=use_local, task_attempt=task_attempt, deadline_ts=deadline_ts,
                model_role=model_role, model_account_override=model_account_override,
            ),
        )
    except Exception as error:  # a failed projection never falls back to the canonical pixels
        propagate_model_error(error)
        log.warning("image preparation failed; images withheld per the image-input mode", exc_info=True)
        return vision_routing.withhold_images(messages, error)


def _send_main_candidate(
    llm: LLMClient, kwargs: Dict[str, Any], *, model: str, use_local: bool, deadline_ts: Optional[float],
    physical_context: Optional[PhysicalAttemptContext],
    candidate_predicate: Optional[Callable[[Any], Any]],
    physical_fit_predicate: Optional[Callable[[Any], Any]] = None,
    send_clock_policy: Any = None, canonical_messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predicate = candidate_predicate if candidate_predicate is not None else physical_fit_predicate
    if candidate_predicate is not None and physical_fit_predicate is not None:
        predicate = lambda request: candidate_predicate(request) is not False and physical_fit_predicate(request) is not False
    binding = (contextlib.nullcontext() if physical_context is None and predicate is None else
               bind_physical_attempt_context(physical_context, candidate_predicate=predicate))
    # Main's clock line per physical send; the consumed one joins ``canonical_messages``.
    with model_concurrency.model_call_scope(model, use_local, deadline_ts), binding, main_send_scope(send_clock_policy, canonical_messages):
        return llm.chat(**kwargs)


def _take_response_facts(usage: Dict[str, Any], msg: Dict[str, Any], accumulated_usage: Dict[str, Any]) -> None:
    model_facts = usage.get("claudexor") or {}
    if model_facts.get("outcome"):
        usage.setdefault("response_status", model_facts["outcome"])
    accumulated_usage["_model_route"] = dict(model_facts.get("route") or {})
    accumulated_usage["_model_substitutions"] = model_facts.get("substituted") or []
    for stale in ("_last_llm_error", "_last_llm_error_kind", "_last_llm_failure", "_acting_route_failure", "_last_llm_retry_same_request",
                  "_last_llm_status_code", "_last_llm_provider_code", "_last_llm_provider_message",
                  "_last_llm_provider_fields", "_last_llm_provider_answer", "_last_llm_provider_message_cut", "_last_llm_resource_refusal",
                  "_last_llm_output_exhausted", "_last_llm_attempt_custody", "_last_llm_size_refusal"):
        accumulated_usage.pop(stale, None)
    receipts = pop_custom_validation_receipts(usage, msg.get("tool_calls") or [])
    accumulated_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
    if receipts:
        accumulated_usage[CUSTOM_RECEIPTS_USAGE_KEY] = receipts


def _own_capture(capture: Any, prior_attempt_id: Optional[str]) -> Any:
    """``capture`` when it is the current logical call's own physical attempt, else None: the attempt captured last
    before the call began (``prior_attempt_id``) stays its own call's fact, so a call that dispatched nothing (a
    fallback refused before its send) carries none rather than borrowing the preceding attempt's."""
    return capture if capture is not None and getattr(capture, "attempt_id", None) != prior_attempt_id else None


def _account_received_error(total: dict, error: ClaudexorModelError, attempt_id: str = "") -> None:
    """Fold the terminal receipt once, including when an exact pause re-reads it."""
    reported = getattr(error, "usage", None)
    if not reported:
        return
    ids = getattr(error, "ledger_attempt_ids", None) or []
    attempt_id = attempt_id or getattr(getattr(error, "physical_attempt_capture", None), "attempt_id", "") or (ids[-1] if ids else "")
    if attempt_id:
        # A ready checkpoint already lists this attempt before it has received any counters.
        if attempt_id in total.get("_counted_error_attempts", ()):
            return
        total["ledger_attempt_ids"] = list(dict.fromkeys([*total.get("ledger_attempt_ids", []), *ids, attempt_id]))
    add_usage(total, reported)
    if attempt_id:
        total.setdefault("_counted_error_attempts", []).append(attempt_id)


def _handle_main_llm_call_exception(
    error: Exception, ctx: _LlmErrorContext, call_identity: Tuple[Any, ...],
) -> bool:
    """Close the exact call and decide whether the attempt loop must stop."""
    _emit_main_llm_call_state(ctx.event_queue, call_identity, "failed",
                              capture=_own_capture(physical_attempt_capture_from_exception(error), ctx.prior_attempt_id))
    ctx.accumulated_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
    refusal = error if isinstance(error, ClaudexorRefusalError) else getattr(error, "received_refusal", None)
    if isinstance(refusal, ClaudexorRefusalError):
        _account_received_error(ctx.accumulated_usage, refusal)
        _record_llm_call_error(refusal, ctx)
        reason = getattr(error, "control_reason", "") or ""
        if reason in ("owner_pause", "owner_pause_authority_unreadable") or reason.startswith("owner_launch_authority_unavailable"):
            capture = getattr(refusal, "physical_attempt_capture", None)
            if capture and refusal.operation_id:
                ctx.accumulated_usage["_pending_transport_outcome"] = {
                    "physical_attempt_id": capture.attempt_id, "operation_id": refusal.operation_id,
                    "model": ctx.model, "model_role": refusal.model_role, "llm_call_id": ctx.llm_call_id,
                    "execution_id": ctx.execution_id, "round_id": ctx.round_id, "round": ctx.round_idx,
                    "attempt": ctx.attempt + 1, "request_ref": (ctx.request_ref or {}).get("manifest_ref"),
                    "paused_receipt_read": True}
        propagate_model_control(error)
        raise refusal
    propagate_model_control(error)  # Task control never enters provider fallback/retry.
    if isinstance(error, ClaudexorModelError):
        _account_received_error(ctx.accumulated_usage, error)
    repeat = TIMEOUT_REPEAT_KEY in ctx.accumulated_usage
    stopped = _record_llm_call_error(error, ctx) or _stop_after_llm_error(ctx)
    if repeat:
        _save_timeout_repeat(ctx.accumulated_usage, failed=True)
    return stopped


def _replace_response_meta(
    target: Optional[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None, msg: Optional[Dict[str, Any]] = None,
) -> None:
    """Replace one caller-owned, attempt-local response fact projection."""
    if target is None:
        return
    target.clear()
    if usage is None or msg is None:
        return
    finish_present, finish_reason = response_finish_reason(usage, msg)
    target.update(finish_reason_present=finish_present, finish_reason=finish_reason,
                  tool_call_count=len(msg.get("tool_calls") or []))
    if usage.get("response_status") is not None:  # a provider-stated terminal (Responses); a Chat reply states none
        target["response_status"] = usage["response_status"]


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
    if isinstance(meta, dict) and "refusal_control_text" in meta:
        return result, dict(meta)
    return str(result or "").strip(), dict(meta) if isinstance(meta, dict) else {}


def _emit_llm_operation(
    event_queue: Optional[queue.Queue], task_id: str, operation_id: str,
    phase: str, task_attempt: Any, execution_id: str, round_id: str,
) -> None:
    timed(
        "event_put", emit_cognitive_operation_event, event_queue, task_id=task_id, operation_id=operation_id,
        phase=phase, kind="llm", task_attempt=task_attempt, execution_id=execution_id,
        round_id=round_id,
    )


def call_llm_with_retry(
    llm: LLMClient, messages: List[Dict[str, Any]], model: str,
    tools: Optional[List[Dict[str, Any]]], effort: str, max_retries: int,
    drive_logs: pathlib.Path, task_id: str, round_idx: int, event_queue: Optional[queue.Queue],
    accumulated_usage: Dict[str, Any], task_type: str = "", use_local: bool = False,
    deadline_ts: Optional[float] = None,
    attempt_cap: Optional[int] = None,
    allow_server_web_search: bool = False,
    physical_context: Optional[PhysicalAttemptContext] = None,
    candidate_predicate: Optional[Callable[[Any], Any]] = None,
    task_attempt: Any = None, response_meta_out: Optional[Dict[str, Any]] = None,
    transport_reserve_sec: Optional[float] = None,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    owner_input_check: Optional[Callable[[], List[str]]] = None,
    model_role: str = "main", model_turn_state: Any = None, model_account_override: Optional[str] = None,
    processing_preference: Optional[str] = None, physical_send_claim: Optional[Callable[[Any], None]] = None,
    send_clock_policy: Any = None, max_tokens: int = MAIN_LOOP_MAX_TOKENS, configured_api_child: bool = False,
    physical_fit_predicate: Optional[Callable[[Any], Any]] = None, checkpoint_call: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Call one model with bounded retries and deadline-aware transport.

    ``max_tokens`` is the LOGICAL caller ceiling (the strict-shrink retry passes the failed
    attempt's sent allowance); the sent value lives on the physical candidate."""
    from ouroboros.model_slots import resolve_processing_preference
    processing_preference = resolve_processing_preference(model_role, override=processing_preference)
    _replace_response_meta(response_meta_out)
    drive_root = pathlib.Path(drive_logs).parent
    for key in (RETRY_WALL_EXHAUSTED_KEY, RETRY_ATTEMPTS_SPENT_KEY, REFUSED_CANDIDATE_KEY, REBOUND_PHYSICAL_CONTEXT_KEY):
        accumulated_usage.pop(key, None)  # last-invocation markers (see the keys)
    execution_id = str(accumulated_usage.setdefault("execution_id", new_execution_id()))
    round_id = f"{execution_id}:round:{round_idx}"
    legacy = accumulated_usage.get(TRANSPORT_DEATHS_KEY)
    if not isinstance(legacy, dict) or legacy.get("round_id") != round_id:
        accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
    record = accumulated_usage.get(TIMEOUT_REPEAT_KEY) or {}
    if timeout_repeat_blocked(accumulated_usage): return None, None
    if record and not record.get("state") and record.get("round_id") != round_id:
        accumulated_usage.pop(TIMEOUT_REPEAT_KEY, None)  # legacy, unrelated round
    context_fit_event_fields = _context_fit_event_fields(accumulated_usage) if physical_context is not None else {}
    transient_budget = _attempt_loop_budget(max_retries, attempt_cap)
    if task_attempt is not None:
        accumulated_usage["_task_attempt"] = task_attempt
    task_attempt = accumulated_usage.get("_task_attempt")
    owner = current_model_wait()
    activity_attempt = task_attempt if task_attempt is not None or not owner or owner.task_id != task_id else owner.attempt
    response_cache_bypass_requested, send_messages = False, None
    restoring = bool(checkpoint_call) and not getattr(checkpoint_call, "forced_reason_code", "")  # a restored end is its round's own call's
    start, transient_budget, max_retries = restored_burst(accumulated_usage if restoring else {}, round_id, model, model_role, transient_budget, max_retries)  # that end's place, in its own burst
    for attempt in range(start, transient_budget):
        for stale in ("_last_llm_failure", "_acting_route_failure"): accumulated_usage.pop(stale, None)  # a pre-send refusal cannot borrow the previous attempt
        if _deadline_not_dispatched(
            deadline_ts, accumulated_usage, drive_logs,
            task_id=task_id, model=model, round_idx=round_idx, round_id=round_id, reserve_sec=transport_reserve_sec,
        ):
            return None, None
        llm_call_id = new_call_id("llm")
        call_identity = (task_id, activity_attempt, llm_call_id, execution_id, round_id, attempt + 1)
        prior = getattr(last_physical_attempt_capture(), "attempt_id", None)  # never this call's context
        call_ids = {"llm_call_id": llm_call_id, "execution_id": execution_id, "round_id": round_id,
                    "round": round_idx, "attempt": attempt + 1}  # this physical call in every record of it below
        request_ref: Dict[str, Any] = {}
        tracker = None
        try:
            _emit_llm_operation(event_queue, task_id, llm_call_id, "started", task_attempt, execution_id, round_id)
            if send_messages is None or TIMEOUT_REPEAT_KEY not in accumulated_usage:  # a timeout repeat resends the projection it timed out on (#1886)
                send_messages = initial_messages if attempt == start and initial_messages is not None else _prepare_main_messages(
                    messages, model=model, model_role=model_role, model_account_override=model_account_override,
                    llm=llm, accumulated_usage=accumulated_usage, drive_root=drive_root, task_id=task_id,
                    event_queue=event_queue, use_local=use_local, task_attempt=task_attempt, deadline_ts=deadline_ts,
                )
            if (accumulated_usage.get(ROUND_STARTED_KEY) or {}).get("round_id") != round_id:
                accumulated_usage[ROUND_STARTED_KEY] = {"round_id": round_id, "at": time.monotonic()}
            _emit_live_log(event_queue, {
                "type": "llm_round_started", "task_id": task_id, "task_type": task_type, **call_ids, "task_attempt": task_attempt,
                "model": model, "reasoning_effort": effort, "use_local": bool(use_local),
            })
            kwargs = {
                "messages": send_messages, "tools": tools, "model": model,
                "chat_conversation_id": task_chat_conversation_id(task_id),
                "model_role": model_role, "model_turn_state": model_turn_state,
                "accepts_typed_refusals": True,
                "model_operation_observer": lambda fact, identity=call_identity, prior=prior: _emit_main_llm_call_state(
                    event_queue, identity, "observed", fact.get("model_activity"), _own_capture(last_physical_attempt_capture(), prior)),
                "model_account_override": model_account_override,
                "processing_preference": processing_preference,
                "reasoning_effort": effort,
                "max_tokens": max_tokens,
                **main_loop_wire_options(model, allow_server_web_search=allow_server_web_search,
                                         bypass_response_cache=response_cache_bypass_requested),
                "caller_deadline_ts": (None if deadline_ts is None
                    else float(deadline_ts) - float(transport_reserve_sec or 0.0)),
                "use_local": use_local,
                "timeout": _main_transport_timeout(model, deadline_ts, reserve_sec=transport_reserve_sec),
            }
            host_step("model_call")  # the round's own facts, fit and send copy, before its first record
            request_ref = persist_observed_call(
                drive_root, task_id=task_id, call_id=f"{llm_call_id}_request", call_type="llm_request",
                payload={
                    "messages": messages,
                    "send_messages": send_messages,  # shares its equal elements with ``messages`` (stored once)
                    "tools": tools or [],
                    "model": model,
                    "reasoning_effort": effort,
                    "max_tokens": max_tokens,  # the logical caller ceiling; the sent allowance is the physical candidate's
                    "use_local": bool(use_local),
                    "allow_server_web_search": bool(allow_server_web_search),
                    "response_cache_bypass_requested": response_cache_bypass_requested,
                },
                manifest={
                    "first_request_at": accumulated_usage.setdefault("first_request_at", utc_now_iso()),
                    **call_ids,
                    "model": model,
                    "reasoning_effort": effort,
                    "response_cache_bypass_requested": response_cache_bypass_requested,
                    **context_fit_event_fields,
                },
                writer=persist_call,
            )
            host_step("request_record")
            if observed_endpoint := accumulated_usage.get("_observed_transport_endpoint"):
                kwargs["observed_endpoint_sha256"] = observed_endpoint
            if _deadline_not_dispatched(
                deadline_ts, accumulated_usage, drive_logs,
                task_id=task_id, model=model, round_idx=round_idx,
                event_queue=event_queue, llm_call_id=llm_call_id,
                task_attempt=task_attempt, execution_id=execution_id,
                round_id=round_id, reserve_sec=transport_reserve_sec,
            ):
                return None, None
            _emit_main_llm_call_state(event_queue, call_identity, "started")
            repeat = TIMEOUT_REPEAT_KEY in accumulated_usage
            if repeat: kwargs["wait_for_resources"] = False
            if restoring and not repeat and (restored := restored_round(accumulated_usage, round_id)):
                kwargs[RESTORED_PROCESSING_KEY] = restored  # its restored attempt's Standard repair and known end
            strict = single_physical_send(lambda reservation: _claim_timeout_repeat(accumulated_usage, call_ids, reservation)) if repeat else contextlib.nullcontext()
            physical_context = refresh_attempt_prefix(physical_context, accumulated_usage, model, use_local)  # a retry after its own answer
            from ouroboros.loop_transport import ReadyRoundAttempts
            tracker = (ReadyRoundAttempts(checkpoint_call, physical_send_claim, {**call_ids, "model_role": model_role, "attempt_budget":
                       transient_budget, "max_retries": max_retries}, request_ref) if checkpoint_call else physical_send_claim)

            def dispatched(capture, identity=call_identity):
                if capture.state == "dispatched":
                    accumulated_usage.pop("_observed_transport_endpoint", None)
                _emit_main_llm_call_state(event_queue, identity, "dispatched", capture=capture)

            with capture_finish_reason() as observation, strict, track_physical_attempts(tracker), observe_dispatch(dispatched):
                msg, usage = _send_main_candidate(
                    llm, kwargs, model=model, use_local=use_local, deadline_ts=deadline_ts,
                    physical_context=physical_context, candidate_predicate=candidate_predicate if attempt == start else None,
                    physical_fit_predicate=physical_fit_predicate, send_clock_policy=send_clock_policy, canonical_messages=messages)
            finish_fields = finish_event_fields(observation, usage, msg)
            mark_last_answer(accumulated_usage)
            host_route = usage.get("model_role_route") or {}
            model, use_local = host_route.get("model", model), host_route.get("use_local", use_local)
            note_prefix_answer(accumulated_usage, usage, model, use_local)  # a reply that read its input moves its route's base
            physical_context = accumulated_usage.pop(REBOUND_PHYSICAL_CONTEXT_KEY, physical_context)
            context_fit_event_fields = _context_fit_event_fields(accumulated_usage) if physical_context is not None else {}
            _take_response_facts(usage, msg, accumulated_usage)
            cost, display_model, provider, cost_estimated = _normalize_usage_cost(usage, model=model, use_local=use_local)
            accumulated_usage["_observed_route"] = observed_route_stamp(usage)
            add_usage(accumulated_usage, usage)
            fold_retrieval_usage(accumulated_usage, usage)
            response_ref = persist_main_response(drive_root, task_id, call_ids, msg, usage, accumulated_usage,
                model=model, display_model=display_model, provider=provider, write=partial(persist_observed_call, writer=persist_call))
            _remember_llm_call(
                accumulated_usage, call_ids={**call_ids, "physical_attempt_id": (usage.get("ledger_attempt_ids") or [None])[-1]}, model=model,
                display_model=display_model, provider=provider, request_ref=request_ref,
                response_ref=response_ref, reported_model=usage.get("resolved_model"), use_local=bool(use_local),
                requested_profile=(usage.get("claudexor") or {}).get("requested_profile", host_route.get("credential_profile_id", model_account_override)),
                observed_route=(usage.get("claudexor") or {}).get("route"),
            )
            category = task_type if task_type in ("evolution", "consciousness", "review", "summarize") else "task"
            emit_llm_usage_event(
                event_queue, task_id, display_model,
                {**usage, "llm_call_id": llm_call_id, "execution_id": execution_id,
                 "round_id": round_id, "round": round_idx},
                cost, category, provider=provider, source="loop", cost_estimated=cost_estimated,
            )
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            if task_type == "presence": accumulated_usage["_presence_pre_dispatch_only"] = False  # a response reached the model
            _replace_response_meta(response_meta_out, usage, msg)
            # A provider-stated terminal that is no result reads as a reply without usable output (``reply_usability``).
            has_text = bool(assistant_answer_text(content).strip())
            if usage_reply_usability(usage, msg) == "unusable" or (
                    not tool_calls and not has_text and refusal_content_text(content) is None):
                event_type, is_provider_glitch, permanent = _record_and_emit_empty_response(
                    usage=usage, msg=msg, accumulated_usage=accumulated_usage,
                    event_queue=event_queue, drive_logs=drive_logs, task_id=task_id,
                    execution_id=execution_id, round_id=round_id, llm_call_id=llm_call_id,
                    round_idx=round_idx, attempt=attempt, model=model, task_type=task_type,
                    content=content, tool_calls=tool_calls, request_ref=request_ref,
                    response_ref=response_ref, transient_budget=transient_budget, prior_attempt_id=prior,
                    context_fit_event_fields=context_fit_event_fields, task_attempt=task_attempt, finish_fields=finish_fields)
                _emit_main_llm_call_state(event_queue, call_identity, "failed")
                if not permanent and attempt < transient_budget - 1 and accumulated_usage.keys().isdisjoint((TRANSPORT_DEATHS_KEY, TIMEOUT_REPEAT_KEY)):
                    if _sleep_within_deadline(min(2.0 ** attempt, _TRANSIENT_BACKOFF_CAP_SEC), deadline_ts):
                        # Only a glitch that is actually retried asks the gateway for a fresh response.
                        response_cache_bypass_requested |= (
                            event_type == "provider_incomplete_response" and not usage.get("provider_error"))
                        continue
                    _emit_retry_deadline_exhausted(
                        drive_logs, task_id=task_id, execution_id=execution_id,
                        round_id=round_id, round_idx=round_idx, attempt=attempt,
                        model=model, error_kind=event_type,
                    )
                if _empty_response_wall_spent(is_provider_glitch, permanent, usage):
                    accumulated_usage[RETRY_WALL_EXHAUSTED_KEY] = True
                return None, cost
            for stale in ("execution_status", "result_status", "reason_code", RETRY_WALL_EXHAUSTED_KEY, RETRY_ATTEMPTS_SPENT_KEY,
                          TRANSPORT_DEATHS_KEY, TIMEOUT_REPEAT_KEY, "_pending_transport_outcome", RESTORED_PROCESSING_KEY, RESTORED_END_KEY):
                accumulated_usage.pop(stale, None)  # a USABLE response closes the round's repeat records
            accumulated_usage["rounds"] = accumulated_usage.get("rounds", 0) + 1
            prompt_cache_ttl, cache_hit_rate, gap_since_prev_round_sec, duration_ms = (
                _record_round_cache_facts(accumulated_usage, usage, round_idx=round_idx, round_id=round_id,
                                          model=display_model, provider=provider))
            ledger_ids = list(usage.get("ledger_attempt_ids") or [])  # the last one carried this response (#807)
            _round_event = {
                "ts": utc_now_iso(), "type": "llm_round", "record_contract_version": RECORD_CONTRACT_VERSION,
                **{key: accumulated_usage[key] for key in ("first_request_at", "first_answer_at")},
                "task_id": task_id, "execution_id": execution_id, **_scope_lineage(),
                "round_id": round_id, "llm_call_id": llm_call_id, "round": round_idx, "model": display_model,
                "reasoning_effort": effort,
                **finish_fields,
                **{key: usage[key] for key in ("effort", "effort_resolution", "request_wire", "claudexor") if key in usage},
                "provider": provider, "source": "loop", "model_category": infer_model_category(display_model),
                **event_token_counts(usage),  # each counter reported, a reported 0, or null (#1578)
                "prompt_cache_ttl": prompt_cache_ttl, "cache_hit_rate": cache_hit_rate,
                "gap_since_prev_round_sec": gap_since_prev_round_sec,
                "duration_ms": duration_ms, "cost_usd": cost, **context_fit_event_fields,
                "ledger_attempt_ids": ledger_ids, "physical_attempt_id": ledger_ids[-1] if ledger_ids else None,
                "request_ref": request_ref.get("manifest_ref") if request_ref else None,
                "response_ref": response_ref.get("manifest_ref") if response_ref else None,
                **round_receipt(),  # what the host did to prepare this round, counted (``llm_observability``)
            }
            _emit_live_log(event_queue, {
                "type": "llm_round_finished", "task_id": task_id, "task_type": task_type, **call_ids,
                "task_attempt": task_attempt, "model": display_model,
                **{key: _round_event[key] for key in (
                    "record_contract_version", "reasoning_effort", "cost_usd", "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
                    "prompt_cache_ttl", "effort", "effort_resolution", "request_wire", "claudexor", "duration_ms") if key in _round_event},
                "response_kind": "tool_calls" if tool_calls else "message",
                "tool_call_count": len(tool_calls), "rounds": accumulated_usage["rounds"],
                "has_text": has_text or bool(refusal_content_text(content)),
            })
            append_jsonl(drive_logs / "events.jsonl", _round_event)
            _emit_main_llm_call_state(event_queue, call_identity, "finished",
                                      capture=_own_capture(last_physical_attempt_capture(), prior))
            return msg, cost
        except (UsageAccountingError, ObservedEndpointChanged):
            _emit_main_llm_call_state(event_queue, call_identity, "failed")
            raise  # Monetary/ledger rails are not provider failures.
        except Exception as e:
            if checkpoint_call and tracker is not None and tracker.reply_source is not None:
                from ouroboros.loop_output_exhaustion import retain_interrupted_main_reply

                retain_interrupted_main_reply(e, tracker.reply_source)
            host_route = getattr(e, "model_role_route", {}) or {}
            model, use_local = host_route.get("model", model), host_route.get("use_local", use_local)
            if isinstance(getattr(e, "route", None), dict):  # a failed operation's route may name no model: the one it asked
                accumulated_usage["_model_route"] = {**e.route, "model": e.route.get("model") or str(model).partition("=")[2]}
            if REBOUND_PHYSICAL_CONTEXT_KEY in accumulated_usage:  # a wait's reprepare bound a new route inside this attempt
                physical_context = accumulated_usage.pop(REBOUND_PHYSICAL_CONTEXT_KEY)
                context_fit_event_fields = _context_fit_event_fields(accumulated_usage)
            if _handle_main_llm_call_exception(e, _LlmErrorContext(
                    task_id=task_id, task_type=task_type, execution_id=execution_id,
                    round_id=round_id, llm_call_id=llm_call_id, round_idx=round_idx,
                    attempt=attempt, model=model, use_local=bool(use_local), request_ref=request_ref,
                    drive_logs=drive_logs, event_queue=event_queue,
                    accumulated_usage=accumulated_usage,
                    context_fit_event_fields=context_fit_event_fields,
                    task_attempt=task_attempt, deadline_ts=deadline_ts,
                    max_retries=max_retries, transient_budget=transient_budget,
                    owner_input_check=owner_input_check,
                    configured_api_child=configured_api_child,
                    requested_profile=host_route.get("credential_profile_id", model_account_override),
                    prior_attempt_id=prior,
                ),
                call_identity,
            ):
                break
    return None, 0.0
